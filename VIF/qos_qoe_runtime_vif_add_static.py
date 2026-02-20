import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler


# ==========================================
# 第一部分：数据加载与预处理
# ==========================================

def load_static_features(file_path):
    """
    读取静态特征文件，并对 label 进行标准化处理以便合并
    """
    try:
        # 读取 CSV，假设分隔符是制表符(\t)或逗号
        # 根据你提供的数据样例，看起来像是制表符分隔，或者是固定宽度的？
        # 这里先尝试自动推断，如果不行请指定 sep='\t'
        df_static = pd.read_csv(file_path, sep='\t')

        # 如果读取出来只有一列，说明分隔符不对，尝试用逗号
        if len(df_static.columns) == 1:
            df_static = pd.read_csv(file_path, sep=',')

        print(f"成功加载静态特征，共 {len(df_static)} 行。")

        # 重命名 label 列为 Domain 以便合并
        df_static.rename(columns={'label': 'Domain'}, inplace=True)

        # [关键] 格式对齐：将下划线替换回点号 (例如 google_com -> google.com)
        # 如果你的 QoS 数据集里的 Domain 是 "google.com" 格式，则需要这步
        # 如果你的 QoS 数据集里的 Domain 也是 "google_com" 格式，则注释掉下面这行
        # df_static['Domain'] = df_static['Domain'].apply(
        #     lambda x: str(x).rsplit('_', 1)[0] + '.' + str(x).rsplit('_', 1)[1] if '_' in str(x) else x)

        return df_static

    except Exception as e:
        print(f"加载静态特征文件失败: {e}")
        return None


def load_and_process_data():
    # 1. 加载三个 NPZ 文件 (用于筛选有效域名)
    npz_files = [
        # "../preprocessed_data/kfp_class_f1_scores.npz",
        # "../preprocessed_data/df_class_f1_scores_avg.npz",
        "../preprocessed_data/star_class_f1_scores.npz"
    ]

    combined_scores = {}
    for file_path in npz_files:
        try:
            with np.load(file_path, allow_pickle=True) as npz_data:
                keys = list(npz_data.keys())
                data_map = None
                if 'arr_0' in keys:
                    content = npz_data['arr_0']
                    if content.ndim == 0:
                        data_map = content.item()
                if data_map is None:
                    data_map = {}
                    for k in keys:
                        val = npz_data[k]
                        if val.ndim == 0:
                            data_map[k] = val.item()
                        else:
                            data_map[k] = val

            for domain, score in data_map.items():
                combined_scores[domain] = 1  # 占位，只需要 Key
        except Exception as e:
            print(f"加载文件 {file_path} 失败: {e}")
            continue

    valid_domains = list(combined_scores.keys())
    df_f1 = pd.DataFrame({"Domain": valid_domains})
    print(f"成功加载有效域名列表 (来自 NPZ)，共 {len(df_f1)} 个。")

    # 2. 加载 QoS CSV
    qos_csv_path = "../preprocessed_data/domain_metrics_full.csv"
    df_qos = pd.read_csv(qos_csv_path)

    # 3. 加载 静态特征 CSV (请在这里填入你的路径)
    # =========================================================================
    static_csv_path = "../preprocessed_data/external_static_features.csv"  # <--- 请替换
    # =========================================================================

    df_static = load_static_features(static_csv_path)

    if df_static is None:
        print("错误：无法继续，静态特征加载失败。")
        return None, None

    # 4. 数据合并: Domain (from NPZ) + QoS + Static
    # 先合并 QoS
    df_step1 = pd.merge(df_f1, df_qos, on="Domain", how="inner")
    print(f"合并 QoS 后数据量: {len(df_step1)}")

    # 再合并 Static (取交集)
    df_final = pd.merge(df_step1, df_static, on="Domain", how="inner")
    print(f"合并 Static 后最终数据量: {len(df_final)}")

    # 5. 提取特征列
    # 排除 Domain 列，保留所有数值列
    feature_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()

    # [优化] 手动剔除已知的高共线性特征 'TCP Handshake'，避免影响整体分析
    if 'TCP Handshake' in feature_cols:
        feature_cols.remove('TCP Handshake')
        print("已自动剔除 'TCP Handshake' (与 TLS Setup 冗余)。")

    # 清洗：填充缺失值
    for col in feature_cols:
        if df_final[col].isnull().any():
            df_final[col] = df_final[col].fillna(df_final[col].median())

    return df_final, feature_cols


# ==========================================
# 第二部分：VIF 分析与可视化 (保持不变)
# ==========================================
def calculate_and_visualize_vif(df, feature_cols):
    print("\n开始执行 VIF 分析...")
    X = df[feature_cols].copy()

    # 剔除常量列
    drop_cols = [col for col in X.columns if X[col].std() == 0]
    if drop_cols:
        print(f"警告: 以下列因方差为0被剔除: {drop_cols}")
        X = X.drop(columns=drop_cols)

    # 处理无穷大值
    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    # 标准化
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 计算 VIF
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_scaled.columns
    vif_data["VIF"] = [variance_inflation_factor(X_scaled.values, i)
                       for i in range(X_scaled.shape[1])]

    vif_data = vif_data.sort_values(by="VIF", ascending=False)

    print("\n=== VIF 分析结果 (Top 10 High Multicollinearity) ===")
    print(vif_data.head(10))
    print("\n=== VIF 分析结果 (Low Multicollinearity - Safe) ===")
    print(vif_data.tail(5))

    # 可视化
    plt.figure(figsize=(10, len(feature_cols) * 0.35))  # 稍微调小一点间距
    colors = ['red' if x > 10 else 'orange' if x > 5 else 'green' for x in vif_data["VIF"]]
    sns.barplot(x="VIF", y="Feature", data=vif_data, palette=colors)
    plt.axvline(x=10, color='r', linestyle='--', label='Threshold (VIF=10)')
    plt.axvline(x=5, color='orange', linestyle='--', label='Threshold (VIF=5)')
    plt.title("VIF Analysis: QoS Metrics vs. Static Features", fontsize=14)
    plt.xlabel("VIF Value", fontsize=12)

    if vif_data["VIF"].max() > 50:
        plt.xscale('log')
        plt.xlabel("VIF Value (Log Scale)", fontsize=12)

    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return vif_data


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    df_final, numeric_cols = load_and_process_data()

    if df_final is not None and len(df_final) > 0:
        vif_df = calculate_and_visualize_vif(df_final, numeric_cols)
        vif_df.to_csv("vif_analysis_with_static.csv", index=False)
        print("\n结果已保存至 'vif_analysis_with_static.csv'")
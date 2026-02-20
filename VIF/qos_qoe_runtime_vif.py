import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler


# ==========================================
# 第一部分：数据加载与预处理 (保持原逻辑)
# ==========================================
def load_and_process_data():
    # 1. 加载三个 NPZ 文件
    npz_files = [
        # "../preprocessed_data/kfp_class_f1_scores.npz",
        # "../preprocessed_data/df_class_f1_scores_avg.npz",
        "../preprocessed_data/star_class_f1_scores.npz"
    ]

    combined_scores = {}

    # --- (这部分代码保持你原本的逻辑不变，用于确定有效域名) ---
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
                if domain not in combined_scores:
                    combined_scores[domain] = []
                if isinstance(score, np.ndarray) and score.size == 1:
                    score = score.item()
                combined_scores[domain].append(score)
        except Exception as e:
            print(f"加载文件 {file_path} 失败: {e}")
            continue

    f1_data = []
    for domain, scores in combined_scores.items():
        if len(scores) > 0:
            f1_data.append({"Domain": domain})  # 这里我们只需要Domain做JoinKey

    df_f1 = pd.DataFrame(f1_data)
    print(f"成功加载有效域名列表，共 {len(df_f1)} 个网站。")

    # 2. 加载 QoS CSV
    csv_path = "../preprocessed_data/domain_metrics_full.csv"
    df_qos = pd.read_csv(csv_path)

    # 3. 数据合并 (Inner Join 确保只分析有对应 F1 Score 的网站)
    df_merged = pd.merge(df_f1, df_qos, on="Domain", how="inner")

    # 4. 提取纯特征列 (排除 Domain 等非数值列)
    #    这里我们自动筛选数值列，但需要排除可能混入的 F1 或 ID 列
    feature_cols = df_merged.select_dtypes(include=[np.number]).columns.tolist()

    # 清洗：填充缺失值
    for col in feature_cols:
        if df_merged[col].isnull().any():
            df_merged[col] = df_merged[col].fillna(df_merged[col].median())

    print(f"合并后用于分析的数据量: {len(df_merged)} 行。")
    return df_merged, feature_cols


# ==========================================
# 第二部分：VIF 分析与可视化
# ==========================================
def calculate_and_visualize_vif(df, feature_cols):
    print("\n开始执行 VIF 分析...")

    # 1. 准备数据 X
    X = df[feature_cols].copy()

    # [关键步骤] 剔除常量列（方差为0），否则会导致奇异矩阵错误
    drop_cols = [col for col in X.columns if X[col].std() == 0]
    if drop_cols:
        print(f"警告: 以下列因方差为0被剔除: {drop_cols}")
        X = X.drop(columns=drop_cols)

    # [关键步骤] 处理无穷大值
    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    # [关键步骤] 数据标准化
    # VIF 计算依赖于特征的线性关系，虽然理论上与缩放无关，但标准化有助于避免数值计算不稳定的问题
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 2. 计算 VIF
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_scaled.columns

    # 这是一个耗时操作，如果特征非常多(>50)，可能需要一点时间
    vif_data["VIF"] = [variance_inflation_factor(X_scaled.values, i)
                       for i in range(X_scaled.shape[1])]

    # 3. 排序 (VIF 越高代表共线性越严重)
    vif_data = vif_data.sort_values(by="VIF", ascending=False)

    # 4. 打印结果
    print("\n=== VIF 分析结果 (Top 10 High Multicollinearity) ===")
    print(vif_data.head(10))
    print("\n=== VIF 分析结果 (Low Multicollinearity - Safe) ===")
    print(vif_data.tail(5))

    # 5. 可视化
    plt.figure(figsize=(10, len(feature_cols) * 0.4))  # 动态调整高度

    # 设定颜色：VIF > 10 为红色(危险)，5-10 为橙色(警告)，< 5 为绿色(安全)
    colors = ['red' if x > 10 else 'orange' if x > 5 else 'green' for x in vif_data["VIF"]]

    sns.barplot(x="VIF", y="Feature", data=vif_data, palette=colors)

    # 添加阈值线
    plt.axvline(x=10, color='r', linestyle='--', label='Threshold (VIF=10)')
    plt.axvline(x=5, color='orange', linestyle='--', label='Threshold (VIF=5)')

    plt.title("Variance Inflation Factor (VIF) Analysis", fontsize=14)
    plt.xlabel("VIF Value (Log Scale recommended if values are huge)", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # 如果有极大的 VIF 值，建议开启对数坐标，否则图很难看
    if vif_data["VIF"].max() > 50:
        plt.xscale('log')
        plt.xlabel("VIF Value (Log Scale)", fontsize=12)

    plt.tight_layout()
    plt.show()

    return vif_data


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 加载数据
    df_final, numeric_cols = load_and_process_data()

    # 2. 执行 VIF 分析
    if len(df_final) > 0:
        vif_df = calculate_and_visualize_vif(df_final, numeric_cols)

        # 3. 保存结果到 CSV，方便你后续发给我讨论
        vif_df.to_csv("vif_analysis_result.csv", index=False)
        print("\nVIF 结果已保存至 'vif_analysis_result.csv'")
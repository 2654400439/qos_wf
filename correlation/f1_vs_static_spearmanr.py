import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ==========================================
# 第二部分：数据加载与预处理
# ==========================================

def load_and_process_data():
    print("开始加载数据...")

    # 1. 加载三个 NPZ 文件 (易识别性指标)
    # 请替换为你真实的路径
    npz_files = [
        "../preprocessed_data/kfp_class_f1_scores.npz",
        "../preprocessed_data/df_class_f1_scores_avg.npz",
        "../preprocessed_data/star_class_f1_scores.npz",
        "../preprocessed_data/finewp_class_f1_scores_avg.npz",
        "../preprocessed_data/rf_class_f1_scores.npz"
    ]

    # 用于存储合并后的分数: {domain: [score1, score2, score3]}
    combined_scores = {}

    for file_path in npz_files:
        try:
            with np.load(file_path, allow_pickle=True) as npz_data:
                keys = list(npz_data.keys())
                data_map = None

                # 尝试提取数据逻辑
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

                if not isinstance(data_map, dict):
                    print(f"警告: 文件 {file_path} 解析后不是字典，跳过。")
                    continue

            # 将提取出的数据存入 combined_scores
            for domain, score in data_map.items():
                if domain not in combined_scores:
                    combined_scores[domain] = []
                if isinstance(score, np.ndarray) and score.size == 1:
                    score = score.item()
                combined_scores[domain].append(score)

        except Exception as e:
            print(f"加载文件 {file_path} 失败: {e}")
            continue

    # 计算 F1_Mean 和 F1_Max
    f1_data = []
    for domain, scores in combined_scores.items():
        if len(scores) > 0:
            f1_data.append({
                "Domain": domain,
                "F1_Mean": np.mean(scores),
                "F1_Max": np.max(scores)
            })

    df_f1 = pd.DataFrame(f1_data)
    print(f"成功加载易识别性数据，共 {len(df_f1)} 个网站。")

    # =======================================================
    # 修改点开始：加载新的 CSV 数据
    # =======================================================
    csv_path = "../preprocessed_data/external_static_features.csv"

    try:
        # 读取 CSV
        # 注意：如果你的文件是用 Tab 分隔的，请加上 sep='\t'，如果是逗号分隔则不需要改
        df_qos = pd.read_csv(csv_path)

        # 1. 重命名列：将 'label' 改为 'Domain' 以便合并
        if 'label' in df_qos.columns:
            df_qos.rename(columns={'label': 'Domain'}, inplace=True)
        else:
            print("警告：CSV中未找到 'label' 列，请检查列名。")

        # 2. 格式对齐：CSV中的域名是 'mzstatic_com'，而 NPZ 通常是 'mzstatic.com'
        # 我们将 CSV 中的下划线替换为点号，以确保能匹配上
        # 如果你的 NPZ 里也是下划线，则注释掉下面这一行
        df_qos['Domain'] = df_qos['Domain'].astype(str).str.replace('_', '_', regex=False)

        # 3. 指定我们关心的数值列（根据你提供的新数据）
        # 排除 label(Domain) 列，只保留数值特征
        metric_cols = ['num_uni_domain', 'num_thi_domain', 'num_img']

        # 确保这些列在 dataframe 中存在，不存在的填0或处理
        available_metrics = [col for col in metric_cols if col in df_qos.columns]

        # 填充缺失值 (保留原有的清洗逻辑)
        for col in available_metrics:
            if df_qos[col].isnull().any():
                median_val = df_qos[col].median()
                df_qos[col] = df_qos[col].fillna(median_val)

        print(f"成功加载指标数据，共 {len(df_qos)} 行。")

        # 4. 合并数据 (Inner Join)
        df_merged = pd.merge(df_f1, df_qos, on="Domain", how="inner")
        print(f"合并后有效数据量: {len(df_merged)} 行。")

        # 如果合并结果为0，可能是域名格式没对上
        if len(df_merged) == 0:
            print("注意：合并后数据量为0！请检查 CSV 和 NPZ 中的域名格式是否一致（例如 'abc.com' vs 'abc_com'）。")

        return df_merged, available_metrics

    except Exception as e:
        print(f"加载 CSV 数据失败: {e}")
        return None, []


# ==========================================
# 第三部分：计算相关性与可视化 (基本保持不变)
# ==========================================

def analyze_and_visualize(df, metric_cols):
    if df is None or len(df) == 0:
        print("数据为空，无法分析")
        return

    # 1. 计算斯皮尔曼相关系数
    target_cols = ["F1_Mean", "F1_Max"]
    correlation_results = []

    for metric in metric_cols:
        # 确保列存在
        if metric not in df.columns:
            continue

        if df[metric].std() == 0:
            continue

        for target in target_cols:
            corr, p_val = spearmanr(df[metric], df[target])
            correlation_results.append({
                "Metric": metric,
                "Target": target,
                "Correlation": corr,
                "P_Value": p_val
            })

    if not correlation_results:
        print("未能计算出任何相关性结果（可能是数据列标准差为0）。")
        return

    df_corr = pd.DataFrame(correlation_results)

    # 2. 准备可视化数据
    pivot_corr = df_corr.pivot(index="Metric", columns="Target", values="Correlation")
    pivot_corr["abs_max"] = pivot_corr["F1_Max"].abs()
    pivot_corr = pivot_corr.sort_values(by="abs_max", ascending=False).drop(columns=["abs_max"])

    # 3. 绘图
    plt.figure(figsize=(8, 6))  # 指标变少了，图的高度可以调小一点

    # 字体设置
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    sns.heatmap(pivot_corr, annot=True, fmt=".2f", cmap="vlag",
                center=0, vmin=-1, vmax=1, linewidths=0.5,
                cbar_kws={"label": "Spearman Correlation Coefficient"})

    plt.title("Correlation: Page Metrics vs. Fingerprintability", fontsize=14, pad=20)
    plt.ylabel("Page Metrics", fontsize=12)
    plt.xlabel("Fingerprintability (F1 Score)", fontsize=12)

    plt.tight_layout()
    plt.show()

    # 4. 打印 Top 结果
    print("\n=== Top 正相关指标 (基于 F1_Max) ===")
    print(df_corr[df_corr["Target"] == "F1_Mean"].sort_values("Correlation", ascending=False).head(5))

    print("\n=== Top 负相关指标 (基于 F1_Max) ===")
    print(df_corr[df_corr["Target"] == "F1_Mean"].sort_values("Correlation", ascending=True).head(5))


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 处理数据
    df_final, metric_columns = load_and_process_data()

    # 2. 分析与绘图
    analyze_and_visualize(df_final, metric_columns)
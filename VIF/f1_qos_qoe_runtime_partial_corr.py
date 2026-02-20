import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_and_process_data():
    print("开始加载数据...")

    # 1. 加载 5 个 NPZ 文件 (包含了所有攻击模型的结果)
    npz_files = [
        "../preprocessed_data/kfp_class_f1_scores.npz",
        "../preprocessed_data/df_class_f1_scores_avg.npz",
        "../preprocessed_data/star_class_f1_scores.npz",
        "../preprocessed_data/finewp_class_f1_scores_avg.npz",
        "../preprocessed_data/rf_class_f1_scores.npz"
    ]

    # 用于存储合并后的分数: {domain: [score1, score2, ...]}
    combined_scores = {}

    for file_path in npz_files:
        try:
            with np.load(file_path, allow_pickle=True) as npz_data:
                keys = list(npz_data.keys())
                data_map = None

                # 策略: 检查是否有一个主键（如 'arr_0'）存储了整个字典
                if 'arr_0' in keys:
                    content = npz_data['arr_0']
                    if content.ndim == 0:
                        data_map = content.item()

                # 如果没有命中 'arr_0'，尝试直接读取所有键
                if data_map is None:
                    data_map = {}
                    for k in keys:
                        val = npz_data[k]
                        # 提取 0-d array 的标量值
                        if val.ndim == 0:
                            data_map[k] = val.item()
                        else:
                            data_map[k] = val

            # 将提取出的数据存入 combined_scores
            if data_map:
                for domain, score in data_map.items():
                    if domain not in combined_scores:
                        combined_scores[domain] = []

                    # 确保分数是标量
                    if isinstance(score, np.ndarray) and score.size == 1:
                        score = score.item()
                    combined_scores[domain].append(score)

        except Exception as e:
            print(f"加载文件 {file_path} 失败: {e}")
            continue

    # [核心修复] 计算 F1_Mean 和 F1_Max
    f1_data = []
    for domain, scores in combined_scores.items():
        if len(scores) > 0:
            f1_data.append({
                "Domain": domain,
                "F1_Mean": np.mean(scores),  # 计算均值
                "F1_Max": np.max(scores)  # 计算最大值
            })

    df_f1 = pd.DataFrame(f1_data)
    print(f"成功加载易识别性数据，共 {len(df_f1)} 个网站。")

    # 2. 加载 QoS CSV
    csv_path = "../preprocessed_data/domain_metrics_full.csv"
    try:
        df_qos = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误: 找不到 QoS 文件 {csv_path}")
        return pd.DataFrame(), []

    # 3. 数据合并 (Inner Join)
    df_merged = pd.merge(df_f1, df_qos, on="Domain", how="inner")

    # 4. 提取纯特征列
    #    这里我们自动筛选数值列
    feature_cols = df_merged.select_dtypes(include=[np.number]).columns.tolist()

    # [重要] 从特征列表中移除 Target 列 (F1_Mean, F1_Max)
    # 否则做偏相关或 VIF 时会把自己和自己做回归
    for target in ['F1_Mean', 'F1_Max']:
        if target in feature_cols:
            feature_cols.remove(target)

    # 清洗：填充缺失值
    for col in feature_cols:
        if df_merged[col].isnull().any():
            df_merged[col] = df_merged[col].fillna(df_merged[col].median())

    print(f"合并后用于分析的数据量: {len(df_merged)} 行。")
    return df_merged, feature_cols


# 假设 df_final 是你包含所有 QoS 指标和 F1 Score 的 DataFrame
# 且 numeric_cols 是你的特征列表

def calculate_partial_correlation(df, x_col, y_col, covar_col):
    """
    手动计算偏相关系数 (Spearman Rank Partial Correlation)
    公式: r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2) * (1 - r_yz^2))
    这里我们需要先转 rank 再算 Pearson，等价于 Spearman Partial
    """
    # 1. 转为 Rank (处理非线性)
    df_rank = df[[x_col, y_col, covar_col]].rank()

    # 2. 计算两两相关系数 (Pearson on Ranks = Spearman)
    corr_mat = df_rank.corr(method='pearson')
    r_xy = corr_mat.loc[x_col, y_col]
    r_xz = corr_mat.loc[x_col, covar_col]
    r_yz = corr_mat.loc[y_col, covar_col]

    # 3. 套用偏相关公式
    # 避免分母为0
    denom = np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
    if denom == 0:
        return 0
    p_corr = (r_xy - r_xz * r_yz) / denom
    return p_corr


def run_size_independence_test(df, metric_cols, target_col="F1_Mean", control_col="Total Transferred Bytes"):
    results = []

    print(f"正在执行偏相关分析，控制变量: {control_col} ...")

    for metric in metric_cols:
        if metric == control_col:
            continue

        # 1. 原始相关性 (Spearman)
        raw_corr, _ = stats.spearmanr(df[metric], df[target_col])

        # 2. 偏相关性 (Control for Size)
        # 确保数据无 NaN
        sub_df = df[[metric, target_col, control_col]].dropna()
        part_corr = calculate_partial_correlation(sub_df, metric, target_col, control_col)

        # 3. 计算保留率 (Retention Rate)
        # 如果原始相关性很小(<0.05)，保留率没有意义
        if abs(raw_corr) > 0.05:
            retention = part_corr / raw_corr
        else:
            retention = 0

        results.append({
            "Metric": metric,
            "Raw Correlation": raw_corr,
            "Partial Correlation (Size-Adjusted)": part_corr,
            "Retention %": retention * 100,
            "Abs_Raw": abs(raw_corr)  # 用于排序
        })

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by="Abs_Raw", ascending=False)

    return df_res


if __name__ == "__main__":
    # 1. 加载数据
    df_final, numeric_cols = load_and_process_data()
    # =================使用示例=================
    # 假设你已经有了 df_final
    # control_col 必须是你的列名，可能是 "Total Transferred Bytes" 或 "Total Bytes"
    df_partial = run_size_independence_test(df_final, numeric_cols, control_col="Total Transferred Bytes")

    # 打印结果看看
    print(df_partial[["Metric", "Raw Correlation", "Partial Correlation (Size-Adjusted)"]].head(10))

    # 保存
    df_partial.to_csv("partial_correlation_results.csv", index=False)
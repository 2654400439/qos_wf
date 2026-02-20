import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
# ==========================================
# 第二部分：数据加载与预处理
# ==========================================

def load_and_process_data():
    # 1. 加载三个 NPZ 文件 (易识别性指标)
    # 请替换为你真实的路径
    npz_files = [
        # "../preprocessed_data/kfp_class_f1_scores.npz",
        # "../preprocessed_data/df_class_f1_scores_avg.npz",
        "../preprocessed_data/star_class_f1_scores.npz"
    ]

    # 用于存储合并后的分数: {domain: [score1, score2, score3]}
    combined_scores = {}

    for file_path in npz_files:
        try:
            # 修改点开始 ==============================================
            # 使用 with 语句加载，确保文件正确关闭
            with np.load(file_path, allow_pickle=True) as npz_data:
                # 策略 A: 检查是否有一个主键（如 'arr_0'）存储了整个字典
                keys = list(npz_data.keys())

                data_map = None

                # 情况 1: 如果保存时是 np.savez('file', my_dict)，数据在 'arr_0' 中
                if 'arr_0' in keys:
                    # 提取 0-d 数组中的对象
                    content = npz_data['arr_0']
                    if content.ndim == 0:
                        data_map = content.item()

                # 情况 2: 如果保存时是 np.savez('file', **my_dict)，每个域名是一个键
                # 或者情况 1 没命中，尝试直接读取所有键
                if data_map is None:
                    # 假设文件里的 keys 就是域名，values 就是分数
                    # 注意：如果值是 0-d 数组，需要 .item() 取出标量
                    data_map = {}
                    for k in keys:
                        val = npz_data[k]
                        # 如果是包含单值的数组，提取出来；如果是数组，保持原样
                        if val.ndim == 0:
                            data_map[k] = val.item()
                        else:
                            data_map[k] = val

                # 简单校验一下提取是否成功
                if not isinstance(data_map, dict):
                    print(f"警告: 文件 {file_path} 解析后不是字典，跳过。Keys: {keys}")
                    continue
            # 修改点结束 ==============================================

            # 将提取出的数据存入 combined_scores
            for domain, score in data_map.items():
                if domain not in combined_scores:
                    combined_scores[domain] = []
                # 处理可能存在的单元素数组情况
                if isinstance(score, np.ndarray) and score.size == 1:
                    score = score.item()
                combined_scores[domain].append(score)

        except Exception as e:
            print(f"加载文件 {file_path} 失败: {e}")
            # 打印更详细的错误堆栈有助于调试，如果需要可以加上 traceback
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

    # 2. 加载 QoS/QoE/Runtime CSV 文件
    # 请确认这个路径是否正确
    csv_path = "../preprocessed_data/domain_metrics_full.csv"
    # 如果你的 CSV 分隔符不是逗号（比如是制表符），请加上 sep='\t'
    df_qos = pd.read_csv(csv_path)

    # ... (后续代码保持不变) ...

    # 3. 数据清洗 (QoS数据)
    numeric_cols = df_qos.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df_qos[col].isnull().any():
            median_val = df_qos[col].median()
            df_qos[col] = df_qos[col].fillna(median_val)

    print(f"成功加载 QoS 数据，共 {len(df_qos)} 行。")

    # 4. 合并数据
    df_merged = pd.merge(df_f1, df_qos, on="Domain", how="inner")
    print(f"合并后有效数据量: {len(df_merged)} 行。")

    return df_merged, numeric_cols


# ==========================================
# 第三部分：计算相关性与可视化
# ==========================================

def analyze_and_visualize(df, metric_cols):
    if df is None or len(df) == 0:
        print("数据为空，无法分析")
        return

    # 1. 计算斯皮尔曼相关系数 (Spearman Correlation)
    # 我们只关心 Metric Columns 与 F1_Mean, F1_Max 之间的关系
    target_cols = ["F1_Mean", "F1_Max"]

    correlation_results = []

    for metric in metric_cols:
        # 跳过完全是常数的列 (方差为0)，否则相关系数无法计算
        if df[metric].std() == 0:
            continue

        for target in target_cols:
            # 计算相关系数和 p-value
            corr, p_val = spearmanr(df[metric], df[target])
            correlation_results.append({
                "Metric": metric,
                "Target": target,
                "Correlation": corr,
                "P_Value": p_val
            })

    df_corr = pd.DataFrame(correlation_results)

    # 2. 准备可视化数据矩阵
    # 我们希望画一个热力图：行是 QoS 指标，列是 F1_Mean 和 F1_Max
    pivot_corr = df_corr.pivot(index="Metric", columns="Target", values="Correlation")

    # 按照 F1_Max 的相关性绝对值大小进行排序，这样最相关的指标会排在前面/后面
    pivot_corr["abs_max"] = pivot_corr["F1_Max"].abs()
    pivot_corr = pivot_corr.sort_values(by="abs_max", ascending=False).drop(columns=["abs_max"])

    # 3. 绘图 (Seaborn Heatmap)
    plt.figure(figsize=(8, 12))  # 高度根据指标数量调整

    # 设置中文字体 (如果有需要，否则保留英文)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 绘制热力图
    # cmap="vlag" 是红蓝配色 (冷暖色)，适合表示正负相关性
    sns.heatmap(pivot_corr, annot=True, fmt=".2f", cmap="vlag",
                center=0, vmin=-1, vmax=1, linewidths=0.5,
                cbar_kws={"label": "Spearman Correlation Coefficient"})

    plt.title("Correlation: Website Metrics vs. Fingerprintability", fontsize=14, pad=20)
    plt.ylabel("Web Performance & Runtime Metrics", fontsize=12)
    plt.xlabel("Fingerprintability (F1 Score)", fontsize=12)

    plt.tight_layout()
    plt.show()

    # 4. 打印 Top 相关性结果
    print("\n=== Top 5 正相关指标 (基于 F1_Max) ===")
    print(df_corr[df_corr["Target"] == "F1_Max"].sort_values("Correlation", ascending=False).head(5))

    print("\n=== Top 5 负相关指标 (基于 F1_Max) ===")
    print(df_corr[df_corr["Target"] == "F1_Max"].sort_values("Correlation", ascending=True).head(5))


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 如果你没有真实数据，取消下面这行的注释先生成假数据测试
    # create_mock_data()

    # 2. 处理数据
    # 请确保将 load_and_process_data 函数中的文件名替换为你真实的文件名
    df_final, metric_columns = load_and_process_data()

    # 3. 分析与绘图
    analyze_and_visualize(df_final, metric_columns)
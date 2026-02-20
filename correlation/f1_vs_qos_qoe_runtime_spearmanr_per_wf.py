import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os

# ==========================================
# 第一部分：配置与辅助函数
# ==========================================
# 设置绘图风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

# 定义指标分组 (映射关系)
METRIC_GROUPS = {
    "Static Stats": [  # 新增的静态资源指标
        'num_uni_domain', 'num_thi_domain', 'num_img'
    ],
    "Web QoS": [
        'HTTP3 Ratio', 'CDN Ratio', 'TLS Setup', 'TCP Handshake',
        'Total Transferred Bytes', 'TTFB', 'DNS Lookup Time',
        'Load Time', 'Request Count', 'Avg Resource Bytes',
        'TLS 1.3 Ratio', 'Connection Reuse Ratio'
    ],
    "Runtime Behavior": [
        'JS Exec Time', 'JS Long Task Count', 'Paint Count',
        'Layout Count', 'Style Recalc'
    ],
    "QoE Metrics": [
        'LCP', 'FCP', 'CLS', 'TTI', 'TBT', 'Speed Index'
    ]
}


def load_npz_scores(file_path):
    """辅助函数：安全加载单个NPZ文件并返回 {domain: score} 字典"""
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
                    data_map[k] = val.item() if val.ndim == 0 else val
            clean_map = {}
            if isinstance(data_map, dict):
                for k, v in data_map.items():
                    score = np.mean(v) if isinstance(v, (np.ndarray, list)) else v
                    clean_map[k] = score
            return clean_map
    except Exception as e:
        print(f"加载文件 {file_path} 失败: {e}")
        return {}


# ==========================================
# 第二部分：数据加载与预处理
# ==========================================

def load_and_process_data():
    # 1. 加载模型分数 (NPZ)
    model_files = {
        "k-FP": "../preprocessed_data/kfp_class_f1_scores.npz",
        "DF": "../preprocessed_data/df_class_f1_scores_avg.npz",
        "STAR": "../preprocessed_data/star_class_f1_scores.npz",
        "FineWP": "../preprocessed_data/finewp_class_f1_scores_avg.npz",
        "RF": "../preprocessed_data/rf_class_f1_scores.npz"
    }

    dfs = []
    for model_name, file_path in model_files.items():
        if os.path.exists(file_path):
            scores_map = load_npz_scores(file_path)
            df_model = pd.DataFrame(list(scores_map.items()), columns=['Domain', f'F1_{model_name}'])
            dfs.append(df_model)
        else:
            print(f"警告: 文件不存在 {file_path}")

    if not dfs: return None, None

    # 合并模型分数
    df_f1 = dfs[0]
    for i in range(1, len(dfs)):
        df_f1 = pd.merge(df_f1, dfs[i], on='Domain', how='inner')
    print(f"模型数据合并完成，共 {len(df_f1)} 个共有网站")

    # 2. 加载 QoS CSV 文件
    qos_csv_path = "../preprocessed_data/domain_metrics_full.csv"
    df_qos = pd.read_csv(qos_csv_path)

    # 简单的格式对齐 (假设之前提到的_转.逻辑还需要，如果不需要可注释)
    if 'label' in df_qos.columns: df_qos.rename(columns={'label': 'Domain'}, inplace=True)
    # df_qos['Domain'] = df_qos['Domain'].astype(str).str.replace('_', '.', regex=False) # 按需保留

    # 3. 加载新的静态特征 CSV 文件
    static_csv_path = "../preprocessed_data/external_static_features.csv"
    if os.path.exists(static_csv_path):
        # 尝试自动推断分隔符，因为你的样本看起来像 Tab 或 Space，但描述说是 csv
        try:
            df_static = pd.read_csv(static_csv_path, sep=None, engine='python')
        except:
            df_static = pd.read_csv(static_csv_path)  # Fallback

        # 重命名 label -> Domain 以便合并
        if 'label' in df_static.columns:
            df_static.rename(columns={'label': 'Domain'}, inplace=True)

        # 严格遵守指令：不进行下划线转换
        print(f"加载静态特征成功，共 {len(df_static)} 行")

        # 先将 Static 和 QoS 合并 (Outer join 以防某些域名只有一种指标)
        # 注意：这里假设 df_qos 和 df_static 的 Domain 格式是一致的
        df_metrics = pd.merge(df_qos, df_static, on='Domain', how='outer')
    else:
        print("警告：未找到静态特征文件，跳过加载。")
        df_metrics = df_qos

    # 4. 数据清洗 (填充所有数值列)
    numeric_cols = df_metrics.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_exclude = ['Rank', 'Label']
    numeric_cols = [c for c in numeric_cols if c not in cols_to_exclude]

    for col in numeric_cols:
        if df_metrics[col].isnull().any():
            median_val = df_metrics[col].median()
            df_metrics[col] = df_metrics[col].fillna(median_val)

    # 5. 最终合并 (Metrics + F1 Scores)
    df_merged = pd.merge(df_f1, df_metrics, on="Domain", how="inner")
    print(f"最终合并有效数据量: {len(df_merged)} 行")

    return df_merged, numeric_cols


# ==========================================
# 第三部分：计算相关性与可视化 (美化版)
# ==========================================

# ==========================================
# 第三部分：计算相关性与可视化 (最终优化版)
# ==========================================

def analyze_attacker_preference(df, all_metric_cols):
    if df is None or len(df) == 0: return

    # 目标列
    target_cols = [c for c in df.columns if c.startswith('F1_')]

    # 1. 计算所有指标的相关性
    correlation_data = []

    # 展平分组字典
    flat_group_metrics = []
    for metrics in METRIC_GROUPS.values():
        flat_group_metrics.extend(metrics)

    # 过滤有效指标
    valid_metrics = [m for m in flat_group_metrics if m in df.columns and df[m].std() != 0]

    for metric in valid_metrics:
        row = {'Metric': metric}
        for target in target_cols:
            corr, _ = spearmanr(df[metric], df[target])
            row[target.replace('F1_', '')] = corr
        correlation_data.append(row)

    df_corr = pd.DataFrame(correlation_data).set_index('Metric')

    # 2. 按照分组顺序重组 DataFrame
    ordered_metrics = []
    group_boundaries = []
    special_boundary = None
    group_labels = []

    current_idx = 0

    for group_name, metrics in METRIC_GROUPS.items():
        existing_metrics = [m for m in metrics if m in df_corr.index]

        if existing_metrics:
            ordered_metrics.extend(existing_metrics)
            count = len(existing_metrics)
            end_idx = current_idx + count

            # 记录标签中心位置
            center_pos = current_idx + count / 2
            group_labels.append((group_name, center_pos))

            # 记录边界
            if group_name == "Static Stats":
                special_boundary = end_idx
            else:
                group_boundaries.append(end_idx)

            current_idx = end_idx

    # 重组数据
    df_corr = df_corr.reindex(ordered_metrics)

    # 3. 绘图设置
    # 动态高度
    fig_height = len(df_corr) * 0.4 + 1.5
    # 宽度设为 9，给左侧留更多空间
    plt.figure(figsize=(9, fig_height))

    # 绘制热力图
    ax = sns.heatmap(df_corr, annot=True, fmt=".2f", cmap="RdBu_r",
                     center=0, vmin=-0.2, vmax=0.2,
                     linewidths=1, linecolor='white',
                     cbar_kws={"label": "Spearman Correlation", "shrink": 0.6, "pad": 0.03})

    plt.title("Attacker Sensitivity: Feature Preference Analysis", fontsize=16, pad=25)
    plt.xlabel("Fingerprinting Method", fontsize=13, labelpad=10)
    plt.ylabel("")

    # 4. 绘制分割线

    # A. 普通组间线 (浅灰色，细线)
    if group_boundaries and group_boundaries[-1] == len(df_corr):
        group_boundaries.pop()
    for y_pos in group_boundaries:
        ax.hlines(y_pos, *ax.get_xlim(), colors='#d9d9d9', linestyles='-', linewidth=1.5)

    # B. 特殊分割线 (Static Stats 下方)
    # 使用黑色实线，线条粗细适中 (linewidth=2.5)
    if special_boundary is not None:
        ax.hlines(special_boundary, *ax.get_xlim(), colors='black', linestyles='-', linewidth=2.5)

    # 5. 添加分组标签
    for name, pos in group_labels:
        # 样式设置
        is_static = (name == "Static Stats")
        # 将静态指标组的标签颜色改为深灰色 (#333333)，比之前的浅灰更明显
        text_color = "#333333" if is_static else "black"
        font_weight = "bold"

        # 调整位置：
        # -0.7 是在 Y轴坐标系(0~1)左侧的偏移量。
        # 如果有些指标名字特别长（如 Total Transferred Bytes），这个值可能需要更小（如 -0.8）
        x_offset = -0.5

        plt.text(x_offset, pos, name,
                 ha='center', va='center',
                 transform=ax.get_yaxis_transform(),
                 fontsize=12, color=text_color, weight=font_weight, rotation=90)

    # 6. 美化 Y 轴具体指标标签
    y_labels = ax.get_yticklabels()
    for label in y_labels:
        label_text = label.get_text()
        if label_text in METRIC_GROUPS.get("Static Stats", []):
            # 将前人指标的具体名字也改为深灰色
            label.set_color("#333333")
            label.set_fontstyle("italic")
            # 保持粗体，或者用 normal 也可以，这里用 normal 区分度更好
            label.set_fontweight("bold")
            # 稍微调小一点字体，体现它是“参考指标”
            label.set_fontsize(10)
        else:
            label.set_color("black")
            label.set_fontweight("bold")

    plt.xticks(rotation=0, fontsize=11)
    # Y轴标签字体大小
    plt.yticks(fontsize=11)

    # 7. 关键修正：大幅增加左侧边距
    # 将 left 增加到 0.35 (35% 的宽度留给左侧标签)，确保左侧的大分类标签不被切掉
    plt.subplots_adjust(left=0.35, bottom=0.1, right=1.0, top=0.9)

    output_path = 'wf_heatmap_final.pdf'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"图表已保存至: {output_path}")
    plt.show()


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    df_final, _ = load_and_process_data()
    # 第二个参数传 None 即可，因为我们在函数内部用了 METRIC_GROUPS 全局变量
    analyze_attacker_preference(df_final, None)
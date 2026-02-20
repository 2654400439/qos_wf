import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
import math
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker


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
            print(f"[{model_name}] 加载成功，共 {len(df_model)} 个网站")
        else:
            print(f"警告: 文件不存在 {file_path}")

    if not dfs:
        return None, None

    df_f1 = dfs[0]
    for i in range(1, len(dfs)):
        df_f1 = pd.merge(df_f1, dfs[i], on='Domain', how='inner')

    csv_path = "../preprocessed_data/domain_metrics_full.csv"
    if not os.path.exists(csv_path):
        print(f"CSV文件不存在: {csv_path}")
        return None, None

    df_qos = pd.read_csv(csv_path)

    numeric_cols = df_qos.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_exclude = ['Rank', 'Label']
    numeric_cols = [c for c in numeric_cols if c not in cols_to_exclude]

    for col in numeric_cols:
        if df_qos[col].isnull().any():
            median_val = df_qos[col].median()
            df_qos[col] = df_qos[col].fillna(median_val)

    df_merged = pd.merge(df_f1, df_qos, on="Domain", how="inner")
    print(f"最终合并有效数据量: {len(df_merged)} 行")

    return df_merged, numeric_cols

# ==========================================
# 第三部分：分组可视化函数 (优化版)
# ==========================================

def format_func(value, tick_number):
    """辅助函数：将大数字转换为 k/M 单位，防止X轴重叠"""
    if value >= 1000000:
        return f'{value / 1000000:.1f}M'
    elif value >= 1000:
        return f'{value / 1000:.0f}k'
    else:
        return f'{value:.0f}'


def plot_grouped_trend_matrix(df, metric_groups, target_col='F1_Mean', num_cols=6):
    """
    绘制分组分箱趋势矩阵图 (优化版：图例右下角，X轴防重叠，Y轴统一)
    """
    # 1. 扁平化指标列表
    all_metrics = []
    metric_to_cat = {}

    # 颜色映射
    cat_palette = {
        'Web QoS': '#1f77b4',  # 蓝色
        'Runtime Behavior': '#ff7f0e',  # 橙色
        'QoE Metrics': '#2ca02c'  # 绿色
    }

    # 按照你给的顺序扁平化
    for cat, metrics in metric_groups.items():
        valid_metrics = [m for m in metrics if m in df.columns]
        all_metrics.extend(valid_metrics)
        for m in valid_metrics:
            metric_to_cat[m] = cat

    if not all_metrics:
        print("没有有效的指标可绘图。")
        return

    # 2. 计算统计指标
    print("正在计算互信息与相关性...")
    mi_scores = mutual_info_regression(df[all_metrics], df[target_col], random_state=42)
    stats_map = {}
    for idx, col in enumerate(all_metrics):
        corr, p = spearmanr(df[col], df[target_col])
        mi = mi_scores[idx]
        stats_map[col] = (corr, mi, p)

    # 3. 准备画布
    num_plots = len(all_metrics)
    num_rows = math.ceil(num_plots / num_cols)

    # 调整画布大小：每行高度稍微增加一点给X轴标签
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3.0, num_rows * 2.5))
    axes = axes.flatten()

    print("正在绘图...")
    for i in range(len(axes)):
        ax = axes[i]

        # --- A. 如果有指标，正常画图 ---
        if i < num_plots:
            col = all_metrics[i]
            cat = metric_to_cat[col]
            color = cat_palette.get(cat, '#333')

            # 绘图
            sns.regplot(
                data=df, x=col, y=target_col, ax=ax,
                x_bins=12,
                scatter_kws={'alpha': 0.5, 's': 20, 'color': 'gray', 'edgecolor': 'none'},
                line_kws={'color': color, 'linewidth': 2.5, 'alpha': 0.9},
                fit_reg=True
            )

            # 标注统计信息
            corr, mi, p = stats_map[col]
            sign = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            title_text = f"{col}\n" + r"$\rho$=" + f"{corr:.2f}{sign} | MI={mi:.2f}"
            ax.set_title(title_text, fontsize=10, fontweight='bold', color=color, pad=8)

            # X轴防重叠处理 (关键修改)
            # 1. 限制刻度数量最多 4 个
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
            # 2. 如果数值很大，使用自定义格式化
            if df[col].max() > 10000:
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))

            # Y轴统一处理 (关键修改)
            # 假设 F1 都在 0.75-1.0 之间，统一范围方便对比斜率
            ax.set_ylim(0.75, 1.02)

            # 坐标轴标签清理
            if i % num_cols == 0:
                ax.set_ylabel("Avg F1 Score", fontsize=9, fontweight='bold')
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])  # 隐藏非第一列的Y轴刻度，让图更干净

            ax.set_xlabel("")
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            sns.despine(ax=ax)

        # --- B. 如果是最后几个空格，处理图例 ---
        else:
            # 关掉坐标轴
            ax.axis('off')

            # 只有在最后一个空格（即右下角）画图例
            # 逻辑：如果是最后一个位置 (axes的最后一个)
            if i == len(axes) - 1:
                legend_elements = [
                    Line2D([0], [0], color=cat_palette['Web QoS'], lw=3, label='Web QoS Trend'),
                    Line2D([0], [0], color=cat_palette['Runtime Behavior'], lw=3, label='Runtime Trend'),
                    Line2D([0], [0], color=cat_palette['QoE Metrics'], lw=3, label='QoE Trend'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Binned Mean'),
                    Line2D([0], [0], color='none', label='Shaded: 95% CI')  # 解释阴影
                ]
                # 在这个子图中心画图例
                ax.legend(handles=legend_elements, loc='center', frameon=False, fontsize=11, title="Legend",
                          title_fontsize=12)

    # 4. 整体布局
    # y=0.98 留出标题空间
    plt.suptitle("Binned Trend Analysis: QoS Metrics vs. Fingerprintability", y=0.98, fontsize=16, fontweight='bold')

    # tight_layout 会自动调整子图间距
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('mi_spearmanr_6cols.pdf', bbox_inches='tight', dpi=300)
    plt.show()


# ==========================================
# 主程序入口 (只需修改最后调用部分)
# ==========================================
if __name__ == "__main__":
    df_final, metric_columns = load_and_process_data()

    if df_final is not None:
        # 1. 自动计算 F1 Mean
        f1_cols = [c for c in df_final.columns if c.startswith('F1_')]
        if 'F1_Mean' not in df_final.columns and f1_cols:
            print(f"正在计算 F1_Mean，基于列: {f1_cols}")
            df_final['F1_Mean'] = df_final[f1_cols].mean(axis=1)

        # 2. 定义分组
        metric_groups = {
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
                'LCP', 'FCP', 'CLS', 'TTI', 'TBT', 'Speed Index'  # 凑够数让右下角留空
            ]
        }

        # 你的指标数量：
        # Web QoS (12) + Runtime (5) + QoE (6) = 23 个指标
        # num_cols = 6
        # 行数 = ceil(23/6) = 4 行
        # 总格子 = 24 个
        # 空余格子 = 24 - 23 = 1 个 (正好是右下角！)

        plot_grouped_trend_matrix(df_final, metric_groups, target_col='F1_Mean', num_cols=6)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.transforms as mtransforms

# --- 1. 学术风格配置 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2  # 坐标轴框线加粗

# 字体稍微调大，增加饱满感
sns.set_context("paper", font_scale=1.6)

STYLE = {
    "grid_alpha": 0.5,  # 网格稍微明显一点
    "grid_style": ":",  # 使用点线，比虚线更精致
    "line_width": 2.5,  # 线条加粗
    "colors": {
        "intra": "#1f77b4",
        "inter": "#d62728",
        "pkts": "#2ca02c",
        "durs": "#ff7f0e"
    }
}


def load_data():
    """读取数据 (保持不变)"""
    try:
        data_a = np.load('../preprocessed_data/df_feature_intra_inter_sim.npz')
        intra, inter = data_a['intra_sims'], data_a['inter_sims']
        data_b = np.load('../preprocessed_data/raw_pcap_pkts_durs.npz')
        pkts, durs = data_b['pkts'], data_b['durs']
        print("Loaded data from .npz files.")
    except Exception as e:
        print(f"Generating dummy data...")
        np.random.seed(42)
        intra = np.clip(np.random.normal(0.75, 0.12, 1000), 0, 1)
        inter = np.clip(np.random.normal(0.35, 0.15, 1000), 0, 1)
        pkts = np.random.lognormal(3.2, 0.5, 1000)
        durs = np.random.lognormal(1.5, 0.6, 1000)
    return intra, inter, pkts, durs


def plot_combined_figure(intra, inter, pkts, durs):
    # --- 2. 创建画布布局 (保持宽长型) ---
    # 稍微缩减一点宽度，增加一点高度，让布局更紧凑
    fig = plt.figure(figsize=(9.5, 5.0), constrained_layout=True)

    # 左右比例：左图稍微宽一点
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.3, 1])
    # 右侧上下间距：极小，让两张图看起来像一个整体
    gs_right = gs[1].subgridspec(2, 1, hspace=0.08)

    ax_cdf = fig.add_subplot(gs[0])
    ax_pkt = fig.add_subplot(gs_right[0])
    ax_dur = fig.add_subplot(gs_right[1])

    # --- 3. 绘制左图：CDF ---
    def get_cdf(data):
        sorted_data = np.sort(data)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        return sorted_data, yvals

    x_intra, y_intra = get_cdf(intra)
    x_inter, y_inter = get_cdf(inter)

    ax_cdf.plot(x_intra, y_intra, label='Intra-class',
                color=STYLE["colors"]["intra"], lw=STYLE["line_width"])
    ax_cdf.plot(x_inter, y_inter, label='Inter-class',
                color=STYLE["colors"]["inter"], lw=STYLE["line_width"], linestyle='--')

    # Gap 标注：位置微调，字体加粗
    gap_y = 0.60
    idx_intra = (np.abs(y_intra - gap_y)).argmin()
    idx_inter = (np.abs(y_inter - gap_y)).argmin()
    x1, x2 = x_inter[idx_inter], x_intra[idx_intra]

    # 箭头加粗
    ax_cdf.annotate('', xy=(x1, gap_y), xytext=(x2, gap_y),
                    arrowprops=dict(arrowstyle='<->', lw=2.0, color='black'))
    # 文字加粗，稍微大一点
    ax_cdf.text((x1 + x2) / 2, gap_y + 0.04, 'Distinguishability\nGap',
                ha='center', va='bottom', fontsize=13, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

    ax_cdf.set_xlabel('Cosine Similarity', fontweight='bold', fontsize=14)
    ax_cdf.set_ylabel('CDF', fontweight='bold', fontsize=14)

    # 关键：稍微保留一点左侧空间 (比如从 -0.05 开始)，防止曲线贴在轴上显得局促
    ax_cdf.set_xlim(-0.02, 1.02)
    ax_cdf.set_ylim(-0.02, 1.02)
    ax_cdf.grid(True, alpha=STYLE["grid_alpha"], linestyle=STYLE["grid_style"])

    # 【改回左上角】：填补空白，但是去掉边框，让它融入背景
    legend = ax_cdf.legend(loc='upper left', frameon=False, fontsize=12, handlelength=1.5)
    # 让图例文字背景稍微白一点，防止遮挡曲线
    legend.get_frame().set_alpha(0.8)

    # --- 4. 绘制右图上部分：Packets ---
    log_pkts = np.log10(pkts)

    # 【关键修改】：找回纹理感
    # bins=40: 增加柱子数量
    # edgecolor='white', linewidth=0.5: 用细白线分割柱子，既有细节又不黑
    sns.histplot(log_pkts, kde=True, ax=ax_pkt, bins=40,
                 color=STYLE["colors"]["pkts"], alpha=0.6,
                 line_kws={'linewidth': 2}, stat='density',
                 element="bars", fill=True, edgecolor='white', linewidth=0.5)

    mean_pkt = np.mean(log_pkts)
    ax_pkt.axvline(mean_pkt, color='k', linestyle=':', lw=2)

    # 标题：加底色，放在右上角
    ax_pkt.text(0.96, 0.85, 'Trace Volume (Log)', transform=ax_pkt.transAxes,
                ha='right', va='center', fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    # 统计值：放在左侧，字号稍微加大
    ax_pkt.text(0.04, 0.85, f'Mean: {10 ** mean_pkt:.0f}', transform=ax_pkt.transAxes,
                fontsize=12, color='#111', ha='left', fontweight='bold')

    ax_pkt.set_ylabel('')
    # 保留 Y 轴刻度，增加信息密度感 (虽然不显示数值)
    ax_pkt.tick_params(axis='y', left=True, labelleft=False, length=3)
    ax_pkt.set_xlabel('')
    ax_pkt.set_xticklabels([])
    ax_pkt.grid(True, alpha=STYLE["grid_alpha"], linestyle=STYLE["grid_style"])

    # --- 5. 绘制右图下部分：Duration ---
    clean_durs = durs[durs < np.percentile(durs, 99)]

    # 同样找回纹理感
    sns.histplot(clean_durs, kde=True, ax=ax_dur, bins=40,
                 color=STYLE["colors"]["durs"], alpha=0.6,
                 line_kws={'linewidth': 2}, stat='density',
                 element="bars", fill=True, edgecolor='white', linewidth=0.5)

    med_dur = np.median(clean_durs)
    ax_dur.axvline(med_dur, color='k', linestyle=':', lw=2)

    ax_dur.text(0.96, 0.85, 'Loading Time', transform=ax_dur.transAxes,
                ha='right', va='center', fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    ax_dur.text(0.04, 0.85, f'Median: {med_dur:.1f}s', transform=ax_dur.transAxes,
                fontsize=12, color='#111', ha='left', fontweight='bold')

    ax_dur.set_ylabel('')
    ax_dur.tick_params(axis='y', left=True, labelleft=False, length=3)
    ax_dur.set_xlabel('Time (s)', fontweight='bold', fontsize=13)
    ax_dur.grid(True, alpha=STYLE["grid_alpha"], linestyle=STYLE["grid_style"])

    # --- 6. 标签 (a)(b)(c) ---
    def label_subplot(ax, label):
        # 字体加大加粗
        ax.text(-0.08, 1.05, label, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='bottom', ha='right')

    label_subplot(ax_cdf, '(a)')
    label_subplot(ax_pkt, '(b)')
    label_subplot(ax_dur, '(c)')

    plt.savefig('combined_figure_dense.pdf', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == "__main__":
    intra, inter, pkts, durs = load_data()
    plot_combined_figure(intra, inter, pkts, durs)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import dpkt
import csv
import os

# 保持与上一段代码一致的风格设置
sns.set_context("paper", font_scale=1.4)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'


def plot_traffic_characteristics_compact(packet_counts, durations):
    """
    绘制紧凑的流量特征概览图：
    1. 左图：包数量分布 (Log Scale)
    2. 右图：加载时长分布
    或者：一张 Joint Plot
    这里我们采用 1x2 Subplots 但做得非常紧凑，适合放在论文底部
    """

    # 创建画布，宽长比适合双栏排版
    fig = plt.figure(figsize=(10, 4))
    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.25)

    # --- Subplot 1: Packet Volume Distribution (Heavy Tail) ---
    ax1 = fig.add_subplot(gs[0])

    # 使用 Log Scale 的直方图 + KDE
    # 注意：对数据取 log10 进行绘图会更直观
    log_packets = np.log10(packet_counts)

    sns.histplot(log_packets, kde=True, stat="density",
                 color="#2ca02c",  # Forest Green，区别于之前的蓝红
                 line_kws={'linewidth': 2}, alpha=0.4, ax=ax1, edgecolor=None)

    ax1.set_xlabel('Log10(Total Packets)', fontweight='bold')
    ax1.set_ylabel('Density', fontweight='bold')
    ax1.set_title('Distribution of Trace Volume', fontsize=12)

    # 添加一些统计标注
    mean_val = np.mean(log_packets)
    ax1.axvline(mean_val, color='black', linestyle='--', alpha=0.6, linewidth=1.5)
    ax1.text(mean_val + 0.1, ax1.get_ylim()[1] * 0.8, f'Mean: {10 ** mean_val:.0f} pkts',
             fontsize=10, color='black')

    # --- Subplot 2: Duration Distribution (QoS Proxy) ---
    ax2 = fig.add_subplot(gs[1])

    # 过滤掉异常大的 Duration (比如超过 60秒的截断，为了绘图好看)
    # 实际论文中不要随便过滤，或者标注截断
    limit_duration = np.percentile(durations, 99)
    clean_durations = durations[durations <= limit_duration]

    sns.histplot(clean_durations, kde=True, stat="density",
                 color="#ff7f0e",  # Orange，代表时间/延迟
                 line_kws={'linewidth': 2}, alpha=0.4, ax=ax2, edgecolor=None)

    ax2.set_xlabel('Loading Time (Seconds)', fontweight='bold')
    ax2.set_ylabel('Density', fontweight='bold')
    ax2.set_title('Distribution of Loading Time', fontsize=12)

    # 标注中位数
    median_val = np.median(clean_durations)
    ax2.axvline(median_val, color='black', linestyle='--', alpha=0.6, linewidth=1.5)
    ax2.text(median_val + 1, ax2.get_ylim()[1] * 0.8, f'Median: {median_val:.2f} s',
             fontsize=10, color='black')

    # --- 整体调整 ---
    plt.tight_layout()
    plt.show()


def plot_traffic_joint_hexbin(packet_counts, durations):
    """
    方案二：更高级的 Joint Plot (Hexbin style)
    展示 Volume vs Duration 的关系，能看出带宽/吞吐量的特征
    """
    # 同样过滤极值以便绘图
    mask = (durations < np.percentile(durations, 99)) & (packet_counts < np.percentile(packet_counts, 99))
    x = durations[mask]
    y = packet_counts[mask]

    # 使用 Seaborn 的 JointGrid
    g = sns.JointGrid(x=x, y=y, height=6, ratio=4)

    # 主图：Hexbin (适合数据点很多的情况，比散点图清晰)
    # 颜色使用深蓝色系
    g.plot_joint(plt.hexbin, gridsize=30, cmap="Blues", mincnt=1, edgecolors='none')

    # 边缘图：KDE
    g.plot_marginals(sns.kdeplot, fill=True, color="#1f77b4", alpha=0.3)

    # 设置坐标轴
    g.ax_joint.set_xlabel('Loading Time (Seconds)', fontweight='bold', fontsize=12)
    g.ax_joint.set_ylabel('Total Packets', fontweight='bold', fontsize=12)

    # 调整 Log Scale (可选，如果 Y 轴跨度太大)
    g.ax_joint.set_yscale('log')
    g.ax_marg_y.set_yscale('log')

    # 添加一个注释，指向“High Throughput”区域
    # (QoS 好的网站：包多但时间短 -> 左上角)
    # g.ax_joint.annotate('High Throughput\n(High QoS)', xy=(...))

    plt.subplots_adjust(top=0.95)
    g.fig.suptitle('Traffic Volume vs. Loading Duration', fontsize=14)
    plt.show()



def count_pcap_packets(file_path):
    count = 0
    try:
        # 1. 必须以 'rb' (二进制只读) 模式打开文件
        with open(file_path, 'rb') as f:
            # 2. 创建 pcap 读取对象
            pcap = dpkt.pcap.Reader(f)

            # 3. 遍历 pcap 对象，每次迭代返回 (时间戳, 数据包内容)
            # 我们只需要统计次数，不需要解析内容
            for ts, buf in pcap:
                count += 1

        print(f"文件 {file_path} 中包含 {count} 个数据包。")
        return count

    except ValueError:
        print("错误：该文件可能不是标准的 pcap 格式 (可能是 pcapng?)")
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
    except Exception as e:
        print(f"读取出错: {e}")


def get_pcap_duration(file_path):
    with open(file_path, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)

        first_ts = None
        last_ts = None

        for ts, buf in pcap:
            if first_ts is None:
                first_ts = ts  # 记录第一个包的时间戳
            last_ts = ts  # 循环结束时，这里就是最后一个包的时间戳

        if first_ts is not None and last_ts is not None:
            duration = last_ts - first_ts
            return duration
        return 0

# --- 模拟数据生成与运行 (你可以替换为你的真实提取代码) ---
if __name__ == "__main__":
    # 都是直接读取每个网站的第一个数据包来获得这些信息吧
    with open('../domain_fine_qoe_pkt.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        domain_list = [row[0] for row in reader]

    pkt_num_list = []
    duration_list = []
    for i in range(1, 11):
        curr_domain = os.listdir(f'../traffic/out_pcap_{i}/out_pcap/')
        for domain in curr_domain:
            if domain in domain_list:
                domain_pcap1 = f'../traffic/out_pcap_{i}/out_pcap/{domain}/round_001.pcap'
                curr_pkt_len1 = count_pcap_packets(domain_pcap1)
                duration1 = get_pcap_duration(domain_pcap1)
                pkt_num_list.append(curr_pkt_len1)
                duration_list.append(duration1)
                domain_pcap2 = f'../traffic/out_pcap_{i}/out_pcap/{domain}/round_002.pcap'
                curr_pkt_len2 = count_pcap_packets(domain_pcap2)
                duration2 = get_pcap_duration(domain_pcap2)
                pkt_num_list.append(curr_pkt_len2)
                duration_list.append(duration2)


    pkts = np.array(pkt_num_list)
    durs = np.array(duration_list)

    np.savez_compressed(
        '../preprocessed_data/raw_pcap_pkts_durs.npz',
        pkts=pkts,
        durs=durs
    )

    print("Generating Compact Histogram Panel...")
    plot_traffic_characteristics_compact(pkts, durs)

    # print("Generating Joint Hexbin Plot...")
    # plot_traffic_joint_hexbin(pkts, durs)
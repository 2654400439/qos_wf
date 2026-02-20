import os
import glob
import numpy as np
import pandas as pd
from scapy.all import PcapReader, IP, IPv6, TCP, UDP
from tqdm import tqdm
import multiprocessing  # [新增] 引入多进程库
import csv

# ================= configuration =================
# 输出 CSV 的路径
OUTPUT_CSV = "../preprocessed_data/traffic_physical_features.csv"
# 进程数：设置为 None 会自动使用所有可用 CPU 核心，或者你可以手动指定，比如 8
NUM_WORKERS = None

with open('../domain_fine_qoe_pkt_mapping.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    tmp = [row for row in reader]
# 排名数据映射 (Placeholder)
# RANK_MAP = {
#     # 'google.com': 1,
# }
RANK_MAP = {k: v for k, v in tmp}


def get_site_rank(domain_label):
    """
    根据域名/标签获取排名的辅助函数
    """
    clean_domain = str(domain_label).strip()
    if clean_domain in RANK_MAP:
        return RANK_MAP[clean_domain]
    if '_' in clean_domain:
        parts = clean_domain.rsplit('_', 1)
        if len(parts) == 2:
            dot_domain = f"{parts[0]}.{parts[1]}"
            if dot_domain in RANK_MAP:
                return RANK_MAP[dot_domain]
    return None


def extract_features_from_pcap(pcap_path):
    """
    核心特征提取逻辑 (保持不变)
    """
    timestamps = []
    packet_sizes = []
    directions = []

    try:
        # scapy 的 PcapReader 在某些多进程环境下可能需要显式关闭，使用 with 语句最安全
        with PcapReader(pcap_path) as packets:
            for pkt in packets:
                if IP in pkt:
                    src_ip = pkt[IP].src
                    length = len(pkt)
                elif IPv6 in pkt:
                    src_ip = pkt[IPv6].src
                    length = len(pkt)
                else:
                    continue

                timestamps.append(float(pkt.time))
                packet_sizes.append(length)
                directions.append(src_ip)

    except Exception as e:
        # 多进程中 print 可能会乱序，但报错还是要看
        # print(f"[Error] 读取 {pcap_path} 失败: {e}")
        return None

    if not timestamps:
        return None

    timestamps = np.array(timestamps)
    timestamps.sort()

    if len(timestamps) > 1:
        iat = np.diff(timestamps)
        mean_iat = np.mean(iat)
        std_iat = np.std(iat)
    else:
        mean_iat = 0.0
        std_iat = 0.0

    burst_sizes = []
    if len(directions) > 0:
        current_burst_bytes = packet_sizes[0]
        prev_src = directions[0]
        for i in range(1, len(directions)):
            curr_src = directions[i]
            curr_len = packet_sizes[i]
            if curr_src == prev_src:
                current_burst_bytes += curr_len
            else:
                burst_sizes.append(current_burst_bytes)
                current_burst_bytes = curr_len
                prev_src = curr_src
        burst_sizes.append(current_burst_bytes)

    burst_sizes = np.array(burst_sizes)
    if len(burst_sizes) > 0:
        mean_burst_size = np.mean(burst_sizes)
        burst_count = len(burst_sizes)
    else:
        mean_burst_size = 0
        burst_count = 0

    total_packets = len(timestamps)

    return {
        "Mean_IAT": mean_iat,
        "Std_IAT": std_iat,
        "Mean_Burst_Size": mean_burst_size,
        "Burst_Count": burst_count,
        "Total_Packets": total_packets
    }


# ==========================================
# [新增] 单个文件处理的包装函数
# ==========================================
def process_single_file(pcap_file):
    """
    这是 Worker 进程执行的函数。
    它负责处理一个文件并返回结果字典（或 None）。
    """
    try:
        filename = os.path.basename(pcap_file)
        # 提取域名逻辑 (根据你的文件名格式调整)
        # domain_label = os.path.splitext(filename)[0]
        # 如果有下划线后缀需要去除，可以在这里加
        # domain_label = domain_label.rsplit('_', 1)[0]
        domain_label = pcap_file.split('/')[-2]

        # 提取特征
        feats = extract_features_from_pcap(pcap_file)

        if feats:
            rank = get_site_rank(domain_label)
            return {
                "Domain": domain_label,
                "Rank": rank,
                **feats
            }
    except Exception as e:
        return None

    return None


def main():
    pcap_files = []
    with open('../domain_fine_qoe_pkt.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        domain_list = [row[0] for row in reader]

    for i in range(1, 11):
        path_list = os.listdir(f'../traffic/out_pcap_{i}/out_pcap/')
        for d in path_list:
            if d in domain_list:
                # pcap_files_curr = glob.glob(os.path.join(f'../traffic/out_pcap_{i}/out_pcap/{d}', "*.pcap"))
                # pcap_files.extend(pcap_files_curr)
                pcap_files.append(f'../traffic/out_pcap_{i}/out_pcap/{d}/round_001.pcap')

    total_files = len(pcap_files)
    print(f"找到 {total_files} 个 PCAP 文件，准备使用多进程处理...")

    results = []

    # 2. 设置多进程池
    # num_processes 如果不传参数，默认使用 os.cpu_count()
    # imap_unordered 比 imap 更快，因为不需要保持结果顺序（我们要存 CSV，顺序无所谓）
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        # 使用 tqdm 包装 pool.imap_unordered 来显示进度条
        # process_single_file 是工作函数，pcap_files 是任务列表
        for res in tqdm(pool.imap_unordered(process_single_file, pcap_files), total=total_files):
            if res is not None:
                results.append(res)

    # 3. 保存结果
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n处理完成！结果已保存至: {OUTPUT_CSV}")
        print(f"共成功提取 {len(df)} 个样本。")
    else:
        print("\n未提取到任何有效数据，请检查路径或文件格式。")


if __name__ == "__main__":
    # Windows/MacOS 下多进程必须在 if __name__ == "__main__": 下运行
    # Linux 下虽然不是强制，但也是好习惯
    main()
import dpkt
import pyshark
import socket
import collections
import numpy as np

def get_flow_id(ip, l4):
    # (src, dst, sport, dport, proto) sorted
    proto = ip.p
    src = socket.inet_ntoa(ip.src)
    dst = socket.inet_ntoa(ip.dst)
    sport = l4.sport
    dport = l4.dport
    if (src, sport, dst, dport, proto) < (dst, dport, src, sport, proto):
        return (src, dst, sport, dport, proto)
    else:
        return (dst, src, dport, sport, proto)


def parse_pcap_by_flow(file_path):
    flows = collections.OrderedDict()  # flow_id: list[ (seq, ts, buf, ip, l4) ]
    with open(file_path, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        for idx, (ts, buf) in enumerate(pcap):
            eth = dpkt.ethernet.Ethernet(buf)
            if not isinstance(eth.data, dpkt.ip.IP):
                continue
            ip = eth.data
            if not hasattr(ip, "data") or not isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                continue
            l4 = ip.data
            flow_id = get_flow_id(ip, l4)
            flows.setdefault(flow_id, []).append((idx, ts, buf, ip, l4))
    return flows


def protocol_label_tcp(flow_pkts):
    found_first = False
    for i, (idx, ts, buf, ip, l4) in enumerate(flow_pkts):
        if hasattr(l4, 'dport') and l4.dport == 443 and l4.data:
            payload = l4.data
            if len(payload) > 0 and payload[0] == 0x17:
                if not found_first:
                    found_first = True
                else:
                    return 1
            elif found_first:
                return 0
    return 0

def process_traffic(file_path, max_len=5000):
    flows = parse_pcap_by_flow(file_path)
    out_pkts = []
    flow_id_to_seq = {}
    for seq, (fid, pkts) in enumerate(flows.items()):
        flow_id_to_seq[fid] = seq
        proto = fid[-1]
        if proto == dpkt.ip.IP_PROTO_UDP:
            proto_mark = 2
        elif proto == dpkt.ip.IP_PROTO_TCP:
            proto_mark = 0  # default 0
            if any(hasattr(l4, 'dport') and l4.dport == 443 for _,_,_,_,l4 in pkts):
                proto_mark = protocol_label_tcp(pkts)
        else:
            proto_mark = -1

        for idx, ts, buf, ip, l4 in pkts:
            direction = 1 if ip.src.startswith(b'\xac\x1f') else (-1 if ip.dst.startswith(b'\xac\x1f') else 0)
            pktlen = len(buf) * direction
            out_pkts.append([
                idx,
                pktlen,
                seq,
                proto_mark
            ])
    out_pkts.sort(key=lambda x: abs(x[0]))

    feat = np.zeros((3, max_len), dtype=np.int32)
    for i, vec in enumerate(out_pkts[:max_len]):
        feat[0, i] = vec[1]
        feat[1, i] = vec[2]
        feat[2, i] = vec[3]
    return feat


import csv
import os
import glob
import numpy as np
from multiprocessing import Pool, Manager
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


# 假设 process_traffic 和 safe_log 已经定义好了
# 如果它们在一个单独的模块里，请 import 进来
# 这里为了代码完整性，我先保留 safe_log 的定义
def safe_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


# ---------------------------------------------------------
# 1. 定义单个任务的处理函数 (Worker Function)
# ---------------------------------------------------------
def worker_task(args):
    """
    这个函数会被多进程调用。
    args 包含：pcap_path (文件路径)
    """
    pcap_path = args

    try:
        # 执行原本的处理逻辑
        # 注意：这里假设 process_traffic 是可调用的
        # 如果 process_traffic 还没定义，请确保代码中有它
        feature = process_traffic(pcap_path, 5000)

        # 确保是浮点型以避免报错
        if feature.dtype != float:
            feature = feature.astype(float)

        # 对第0行执行 safe_log
        feature[0] = safe_log(feature[0])

        # 获取标签 (原代码逻辑：d 也就是倒数第二级目录名)
        # 路径结构: .../out_pcap/{domain}/{filename}.pcap
        label_str = pcap_path.split('/')[-2]

        return feature, label_str
    except Exception as e:
        # 捕获异常，防止单个文件损坏导致整个进程崩掉
        print(f"Error processing {pcap_path}: {e}")
        return None


# ---------------------------------------------------------
# 2. 主流程
# ---------------------------------------------------------
def main():
    # --- 配置路径 ---
    domain_csv_path = '../iwqos_project/domain_fine_qoe_pkt.csv'
    base_traffic_path = '../traffic'
    output_filename = '../preprocessed_data/star_traffic_dataset.npz'

    # --- 1. 读取域名白名单 ---
    print("Loading domain list...")
    with open(domain_csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # 假设 csv 第一列是域名
        domain_set = set(row[0] for row in reader)  # 使用 set 加速查找

    # --- 2. 扫描所有待处理的文件 ---
    print("Scanning files...")
    all_pcap_tasks = []

    for i in range(1, 11):
        # 构建当前批次的根目录
        current_base = os.path.join(base_traffic_path, f'out_pcap_{i}', 'out_pcap')

        if not os.path.exists(current_base):
            continue

        # 获取该目录下的所有域名文件夹
        path_list = os.listdir(current_base)

        for d in path_list:
            # 检查是否在白名单中
            if d in domain_set:
                domain_dir = os.path.join(current_base, d)
                # 查找该域名下的所有 pcap
                pcap_files = glob.glob(os.path.join(domain_dir, "*.pcap"))

                # 将每个文件路径作为一个任务添加到列表中
                for p in pcap_files:
                    all_pcap_tasks.append(p)

    total_files = len(all_pcap_tasks)
    print(f"Total files to process: {total_files}")

    # --- 3. 多进程处理 ---
    # 根据 CPU 核心数设定进程数，通常设为 os.cpu_count() 或者稍微少一点
    num_processes = min(os.cpu_count(), 32)

    results_features = []
    results_labels_str = []

    print(f"Starting multiprocessing with {num_processes} workers...")

    # 使用 Pool 开启多进程
    with Pool(processes=num_processes) as pool:
        # imap_unordered 会稍微快一点，且能配合 tqdm 实时显示进度
        # 只要有任何一个进程完成，就会 yield 结果
        iterator = pool.imap_unordered(worker_task, all_pcap_tasks, chunksize=10)

        for res in tqdm(iterator, total=total_files, desc="Processing Traffic"):
            if res is not None:
                feat, lab = res
                results_features.append(feat)
                results_labels_str.append(lab)

    # --- 4. 数据转换与标签编码 ---
    print("Encoding labels and stacking arrays...")

    # 转换为 numpy 数组
    # 假设每个 feature 的 shape 是一样的 (例如 [Row, Col])
    # 这会形成 (N_samples, Row, Col)
    X_data = np.array(results_features)

    # 标签编码
    le = LabelEncoder()
    y_data = le.fit_transform(results_labels_str)

    # 获取映射关系: 哪个数字对应哪个域名
    # 格式建议： {'google.com': 0, 'youtube.com': 1, ...}
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    print(f"Final data shape: {X_data.shape}")
    print(f"Labels shape: {y_data.shape}")

    # --- 5. 保存结果 ---
    print(f"Saving to {output_filename}...")
    np.savez(output_filename,
             traffic=X_data,
             label=y_data,
             label_mapping=label_mapping)

    print("Done!")


if __name__ == '__main__':
    main()
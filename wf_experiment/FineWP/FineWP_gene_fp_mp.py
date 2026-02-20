import csv
from collections import Counter
import numpy as np
from scipy.stats import kurtosis
from tqdm import trange, tqdm
import multiprocessing as mp
from scapy.all import rdpcap, wrpcap, IP, IPv6, TCP



def process_pcap(file_path):
    all_packets = rdpcap(file_path)
    trace = []

    for pkt in all_packets:
        if not pkt.haslayer(TCP):
            continue
        if pkt.haslayer(IP):
            pass
        elif pkt.haslayer(IPv6):
            pass
        else:
            continue

        sport = pkt[TCP].sport
        length = pkt[IP].len

        if sport == 443:
            trace.append(length)
        else:
            trace.append(-length)

    return trace


def process_file(pcap_file):
    # 你的原始特征提取逻辑
    sequence = process_pcap(pcap_file)

    U0 = [item if item > 0 else 0 for item in sequence]
    for i in range(len(U0) - 1):
        U0[i+1] = U0[i+1] + U0[i]

    counter = dict(Counter(U0))
    A = {key: value for key, value in counter.items() if value >= 4}
    S = {key: U0.index(key) for key in A.keys()}
    E = {key: len(U0) - 1 - U0[::-1].index(key) for key in A.keys()}

    si = list(S.values())
    ei = list(E.values())
    ui = list(S.keys())

    hp_b = 29
    hp_d = 77
    hp_k = 3  # 或者试试12


    B = [(si[i], ei[i], ui[i]) for i in range(len(si))]
    loc = 0
    for i in range(len(B)):
        if B[i][0] > 20:
            loc = i
            break
    B = B[loc: loc+hp_k]
    SF = U0[hp_b: hp_d]

    a_ = sequence
    u_ = [item for item in a_ if item < 0]
    d_ = [item for item in a_ if item > 0]

    try:
        ST = [len(a_), len(u_), len(d_), float(np.std(np.array(u_))), min(u_), float(kurtosis(u_))]
    except ValueError:
        # 返回特殊标志，主进程可打印日志或报错
        return None

    finewp_feature = [item for sublist in B for item in sublist] + SF + ST
    finewp_feature.append(pcap_file.split('/')[-2])
    return finewp_feature

if __name__ == '__main__':
    import os
    import glob
    mp.set_start_method('spawn', force=True)  # 显式指定使用 spawn 模式
    # file_list 应该已经在你的作用域
    output_file = '../preprocessed_data/finewp_features_k_3.csv'

    file_list = []

    with open('../iwqos_project/domain_fine_qoe_pkt.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        domain_list = [row[0] for row in reader]

    for i in range(1, 11):
        path_list = os.listdir(f'../traffic/out_pcap_{i}/out_pcap/')
        for d in path_list:
            if d in domain_list:
                pcap_files_curr = glob.glob(os.path.join(f'../traffic/out_pcap_{i}/out_pcap/{d}', "*.pcap"))
                file_list.extend(pcap_files_curr)

    # 用mp.Pool并行处理
    # with mp.Pool(processes=30) as pool:
    #     features = list(pool.map(process_file, file_list))
    features = []

    with mp.Pool(processes=30) as pool:
        for result in tqdm(
                pool.imap_unordered(process_file, file_list),
                total=len(file_list),
                desc="Processing files"
        ):
            if result is not None:
                features.append(result)

    # 过滤掉None（出错/异常的结果）
    features = [f for f in features if f is not None]

    # 主进程统一写入csv
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for feat in features:
            writer.writerow(feat)

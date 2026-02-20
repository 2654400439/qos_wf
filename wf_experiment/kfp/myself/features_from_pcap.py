# standard variant

# Taken from WTF-PAD or Front/Glue, modified and fixed for non-Tor cells (with real sizes)
# Added size features

import numpy as np
import math
import sys
import traceback

from pathlib import Path

import dpkt
import socket

import multiprocessing as mp
import os
import glob
from tqdm import tqdm

def parse_pcap_file(pcap_file_path):
    """使用 dpkt 解析 pcap 文件，提取时间戳和包大小"""
    packets_data = []

    with open(pcap_file_path, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)

        for timestamp, buf in pcap:
            # 解析以太网帧
            eth = dpkt.ethernet.Ethernet(buf)

            # 只处理 IP 数据包
            if isinstance(eth.data, dpkt.ip.IP):
                ip = eth.data

                # 获取包大小（包括以太网头部）
                packet_size = len(buf)

                # 判断方向（简化处理：根据源 IP 判断）
                # 这里需要根据你的具体需求调整方向判断逻辑
                src_ip = socket.inet_ntoa(ip.src)
                direction = -1 if src_ip.startswith('192.168') else 1  # 示例逻辑

                packets_data.append(['unknown', timestamp, direction * packet_size])

    return packets_data


def average(array):
    if array is None or len(array) == 0:
        return 0
    return np.average(array)


def array_to_fix_size(array, length, pad_with=0):
    if len(array) < length:
        array.extend([pad_with] * (length - len(array)))
    elif len(array) > length:
        array = array[:length]
    return array


def split_in_chunks(array, num_splits):
    avg = len(array) / float(num_splits)
    out = []
    last = 0.0
    while last < len(array):
        out.append(array[int(last): int(last + avg)])
        last += avg
    return out


def split_incoming_outgoing(data):
    incoming = []
    outgoing = []
    for p in data:
        isIncoming = (p[2] < 0)  # p[2] is size
        if isIncoming:
            incoming.append(p)
        else:
            outgoing.append(p)
    return incoming, outgoing


def get_packet_inter_times(data):
    if len(data) == 0:
        return [0]
    times = [x[1] for x in data]
    result = []
    for elem, next_elem in zip(times, times[1:] + [times[0]]):
        result.append(next_elem - elem)
    return result[:-1]


def add_intertimes_stats(features, data, incoming, outgoing):
    # statistics about the inter-packet durations

    def add_stats(trace, prefix=''):
        if trace is not None and len(trace) > 0:
            features['intertime_' + prefix + 'max'] = max(trace)
            features['intertime_' + prefix + 'avg'] = average(trace)
            features['intertime_' + prefix + 'std'] = np.std(trace)
            features['intertime_' + prefix + 'p75'] = np.percentile(trace, 75)
        else:
            features['intertime_' + prefix + 'p25'] = 0
            features['intertime_' + prefix + 'p50'] = 0
            features['intertime_' + prefix + 'p75'] = 0
            features['intertime_' + prefix + 'p100'] = 0

    incoming_intertimes = get_packet_inter_times(incoming)
    outgoing_intertimes = get_packet_inter_times(outgoing)
    all_intertimes = get_packet_inter_times(data)

    add_stats(incoming_intertimes, 'incoming_')
    add_stats(outgoing_intertimes, 'outgoing_')
    add_stats(all_intertimes, '')


def add_time_percentiles(features, data, incoming, outgoing):
    # percentiles about the times in which packets where sent/received

    def add_percentiles(trace, prefix=''):
        if trace is not None and len(trace) > 0:
            features['time_' + prefix + 'p25'] = np.percentile(trace, 25)
            features['time_' + prefix + 'p50'] = np.percentile(trace, 50)
            features['time_' + prefix + 'p75'] = np.percentile(trace, 75)
            features['time_' + prefix + 'p100'] = np.percentile(trace, 100)
        else:
            features['time_' + prefix + 'p25'] = 0
            features['time_' + prefix + 'p50'] = 0
            features['time_' + prefix + 'p75'] = 0
            features['time_' + prefix + 'p100'] = 0

    incoming_times = [x[1] for x in incoming]
    outgoing_times = [x[1] for x in outgoing]
    times = [x[1] for x in data]

    add_percentiles(incoming_times, 'incoming_')
    add_percentiles(outgoing_times, 'outgoing_')
    add_percentiles(times, '')

    features['times_sum'] = sum(times)


def add_counts_in_out_last_first_30(features, data):
    # counts (incoming, outgoing) packets in the (first, last) 30 packets

    first30 = data[:30]
    last30 = data[-30:]
    first30in = []
    first30out = []
    for p in first30:
        if p[2] < 0:  # incoming
            first30in.append(p)
        if p[2] >= 0:
            first30out.append(p)
    last30in = []
    last30out = []
    for p in last30:
        if p[2] < 0:  # incoming
            last30in.append(p)
        if p[2] >= 0:
            last30out.append(p)

    features['f30_n_incoming'] = len(first30in)
    features['f30_n_outgoing'] = len(first30out)
    features['l30_n_incoming'] = len(last30in)
    features['l30_n_outgoing'] = len(last30out)


def add_outgoing_concentrations_stats(features, data):
    # concentration of outgoing packets in chunks of 20 packets

    chunks = [data[x: x + 20] for x in range(0, len(data), 20)]
    concentrations = []
    for item in chunks:
        c = 0
        for p in item:
            if p[2] >= 0:  # outgoing packets
                c += 1
        concentrations.append(c)

    concentrations = array_to_fix_size(concentrations, 40)

    features['outgoing_concentrations_std'] = np.std(concentrations)
    features['outgoing_concentrations_mean'] = average(concentrations)
    features['outgoing_concentrations_p50'] = np.percentile(concentrations, 50)
    features['outgoing_concentrations_min'] = min(concentrations)
    features['outgoing_concentrations_max'] = max(concentrations)

    i = 0
    while i < len(concentrations):
        features['outgoing_concentrations_' + str(i)] = concentrations[i]
        i += 1

    # Same think, but for trace divided in 70 fixed chunks

    outgoing_concentrations_70 = [
        sum(x) for x in split_in_chunks(concentrations, 70)]

    i = 0
    while i < len(outgoing_concentrations_70):
        features['outgoing_concentrations_70_' +
                 str(i)] = outgoing_concentrations_70[i]
        i += 1

    features['outgoing_concentrations_70_sum'] = sum(
        outgoing_concentrations_70)


def add_delta_rates_stats(features, data):
    # Average number packets sent and received per second

    last_time = data[-1][1]
    last_second = math.ceil(last_time)

    count_per_sec = []
    for sec in range(1, int(last_second) + 1):
        count = 0
        for p in data:
            if p[1] <= sec:  # p[1] is packet time
                count += 1
        count_per_sec.append(count)

    count_per_sec = array_to_fix_size(count_per_sec, 10)

    delta_count_per_sec = [0]  # first difference is 0
    i = 1
    while i < len(count_per_sec):
        diff = count_per_sec[i] - count_per_sec[i - 1]
        delta_count_per_sec.append(diff)
        i += 1

    features['delta_rate_avg'] = average(delta_count_per_sec)
    features['delta_rate_std'] = np.std(delta_count_per_sec)
    features['delta_rate_p50'] = np.percentile(delta_count_per_sec, 50)
    features['delta_rate_min'] = min(delta_count_per_sec)
    features['delta_rate_max'] = max(delta_count_per_sec)

    i = 1
    while i < len(delta_count_per_sec):
        features['delta_rate_' + str(i)] = delta_count_per_sec[i]
        i += 1

    # Same thing, but trace divided in 20 fixed chunks

    delta_counts_20 = [sum(x)
                       for x in split_in_chunks(delta_count_per_sec, 20)]

    i = 0
    while i < len(delta_counts_20):
        features['delta_rates_20_' + str(i)] = delta_counts_20[i]
        i += 1

    features['delta_rates_20_sum'] = sum(delta_counts_20)


def add_average_pkt_ordering(features, data):
    # counts the cumulative number of (incoming, outgoing) packet for a given time

    def add_stats(trace, suffix=''):
        if len(trace) == 0:
            features['order_avg' + suffix] = 0
            features['order_std' + suffix] = 0
        else:
            features['order_avg' + suffix] = average(trace)
            features['order_std' + suffix] = np.std(trace)

    c_out = 0
    c_in = 0
    outgoing = []
    incoming = []
    for p in data:
        if p[2] >= 0:  # outgoing
            outgoing.append(c_out)
            c_out += 1
        if p[2] < 0:
            incoming.append(c_in)
            c_in += 1

    add_stats(outgoing, '_out')
    add_stats(incoming, '_in')


def extract_features(data, max_size=244):
    features = dict()

    if len(data) == 0:
        return array_to_fix_size([], max_size, pad_with=('*', 0))

    def quic_to_1(s):
        if s == 'quic':
            return 1
        return 0

    data = [[quic_to_1(p[0]), p[1], p[2]] for p in data]

    total_number_of_packets = len(data)
    total_number_of_quic_pkts = sum([x[0] for x in data])

    features['quic_ratio'] = 0  # total_number_of_quic_pkts / total_number_of_packets

    incoming, outgoing = split_incoming_outgoing(data)
    features['n_incoming'] = len(incoming)
    features['n_outgoing'] = len(outgoing)
    features['n_total'] = len(data)
    features['%_in'] = len(incoming) / float(len(data))
    features['%_out'] = len(outgoing) / float(len(data))

    features['bytes_incoming'] = sum([d[2] for d in incoming])
    features['bytes_outgoing'] = sum([d[2] for d in outgoing])
    features['bytes_total'] = features['bytes_incoming'] + features['bytes_outgoing']
    if features['bytes_total'] > 0:
        features['bytes_%_in'] = features['bytes_incoming'] / float(features['bytes_total'])
        features['bytes_%_out'] = features['bytes_outgoing'] / float(features['bytes_total'])
    else:
        features['bytes_%_in'] = 0
        features['bytes_%_out'] = 0

    add_intertimes_stats(features, data, incoming, outgoing)
    add_time_percentiles(features, data, incoming, outgoing)
    add_counts_in_out_last_first_30(features, data)
    add_average_pkt_ordering(features, data)

    add_outgoing_concentrations_stats(features, data)

    add_delta_rates_stats(features, data)

    # added size features; TLS max is -16K +16k
    incoming_sizes = [-x[2] for x in incoming]
    bins = np.linspace(0, 16 * 1024, 50)
    hist, bin_edges = np.histogram(incoming_sizes, bins=bins, density=False)

    i = 0
    while i < len(hist):
        features['hist_' + str(round(bin_edges[i]))] = hist[i]
        i += 1

    # unmap feature dictionnary for padding
    tuples = [(k, v) for k, v in features.items()]

    features = array_to_fix_size(tuples, max_size, pad_with=('*', 0))

    return features


def trace_starts_at_time0(X):
    if len(X) == 0:
        return X

    t0 = X[0][1]
    i = 0
    while i < len(X):
        X[i][1] -= t0
        i += 1

    return X


def build_from_pcap(pcap_file_path, defense=None):
    """直接从 pcap 文件构建特征数据集"""

    # 解析 pcap 文件
    raw_data = parse_pcap_file(pcap_file_path)

    # 应用防御机制（如果需要）
    if defense is not None:
        raw_data, _, _ = defense(raw_data)

    # 时间归一化
    raw_data = trace_starts_at_time0(raw_data)

    # 提取特征
    features = extract_features(raw_data)

    # 返回特征和文件名作为标签
    return {
        'feature_names': [x[0] for x in features],
        'feature_values': [x[1] for x in features],
        'label': Path(pcap_file_path).parent.name  # 使用文件名作为标签
    }


def process_single_pcap(args):
    """处理单个 pcap 文件的包装函数，用于多进程调用"""
    pcap_file, defense = args
    return build_from_pcap(pcap_file, defense)


def process_multiple_pcaps_parallel(pcap_directory, output_name, defense=None, n_jobs=20):
    """并行处理目录下的所有 pcap 文件，并生成一个统一的 .npy 文件"""

    # # 获取所有 pcap 文件
    # pcap_files = glob.glob(os.path.join(pcap_directory, "*.pcap"))
    # 在这里直接加载pcap文件路径
    pcap_files = []
    for i in range(1, 11):
        path_list = os.listdir(f'../traffic/out_pcap_{i}/out_pcap/')
        for d in path_list:
            pcap_files_curr = glob.glob(os.path.join(f'../traffic/out_pcap_{i}/out_pcap/{d}', "*.pcap"))
            pcap_files.extend(pcap_files_curr)

    if not pcap_files:
        print("未找到任何 .pcap 文件")
        return

    # 准备多进程参数
    args_list = [(pcap_file, defense) for pcap_file in pcap_files]

    if n_jobs is None:
        n_jobs = mp.cpu_count()

    # 使用多进程池并行处理
    # with mp.Pool(processes=n_jobs) as pool:
    #     results = pool.map(process_single_pcap, args_list)
    with mp.Pool(processes=n_jobs) as pool:
        # 使用 imap 而不是 map，这样可以迭代获取结果
        results = []
        # tqdm 会显示进度条
        for result in tqdm(pool.imap(process_single_pcap, args_list),
                           total=len(args_list),
                           desc="Processing PCAP files"):
            results.append(result)

    # 合并所有结果
    feature_names = results[0]['feature_names']  # 假设所有样本的特征名相同
    features_list = [result['feature_values'] for result in results]
    labels_list = [result['label'] for result in results]

    print(labels_list)

    # 构造最终的输出结构
    final_result = {
        'feature_names': feature_names,
        'features': features_list,
        'labels': labels_list
    }

    # 保存结果到 .npy 文件
    np.save(output_name, final_result)
    print(f"所有特征已保存到: {output_name}")


def sign(x):
    if x >= 0:
        return 1
    return -1


if __name__ == "__main__":
    process_multiple_pcaps_parallel('none', 'kfp_features_all.npy')

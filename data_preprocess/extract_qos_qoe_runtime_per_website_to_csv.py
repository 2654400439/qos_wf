import csv
import os
import json
from tqdm import tqdm
import numpy as np


def main_result():
    # ================= 1. 准备域名标签 (Domain Labels) =================
    print("Loading domain lists...")
    with open('../domain_fine_qoe_pkt.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        # 这里的 domain_list 其实只用于确定哪些域名是合法的，或者你可以直接依赖 1800 列表
        all_valid_domains = set([row[0] for row in reader])

    with open('../url_list_1800.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        # 注意：这里保持和你原代码一致的处理逻辑
        domain_1800 = [row[0].split('//')[1].replace('.', '_') for row in reader]

    # 划分列表
    domain_1800_1 = set(domain_1800[:600])
    domain_1800_2 = set(domain_1800[600:1200])
    domain_1800_3 = set(domain_1800[1200:])

    # 建立标签映射表 (Domain -> Label)
    # 这样在循环中查标签是 O(1) 复杂度，不用每次都去遍历列表
    domain_label_map = {}

    # 只需要处理我们在 domain_fine_qoe_pkt.csv 里见过的域名
    # 同时也符合 Top/Torso/Tail 的分类
    for d in all_valid_domains:
        if d in domain_1800_1:
            domain_label_map[d] = 'Top'
        elif d in domain_1800_2:
            domain_label_map[d] = 'Torso'
        elif d in domain_1800_3:
            domain_label_map[d] = 'Tail'
        # else: 'error' 的就不管了，不录入 map 就不会处理

    # ================= 2. 定义数据存储结构 =================
    # 定义所有需要的指标列名
    qos_keys = [
        'DNS Lookup Time', 'TCP Handshake', 'TLS Setup', 'TTFB', 'Load Time',
        'Request Count', 'Total Transferred Bytes', 'Avg Resource Bytes',
        'HTTP3 Ratio', 'TLS 1.3 Ratio', 'Connection Reuse Ratio', 'CDN Ratio'
    ]
    runtime_keys = [
        'JS Exec Time', 'JS Long Task Count', 'Layout Count', 'Paint Count', 'Style Recalc'
    ]
    qoe_keys = [
        'LCP', 'FCP', 'CLS', 'TTI', 'TBT', 'Speed Index'
    ]

    # 总表头
    headers = ['Domain', 'Label'] + qos_keys + runtime_keys + qoe_keys

    # 主数据字典: { 'domain_name': { 'DNS...': 100, 'LCP': 2000... } }
    all_samples = {}

    # 初始化辅助函数
    def get_sample_entry(domain_name):
        domain_name = domain_name.replace('.', '_')
        # 如果这个域名不在我们关心的列表里（Top/Torso/Tail），直接跳过
        if domain_name not in domain_label_map:
            return None

        if domain_name not in all_samples:
            # 初始化一个空行，所有数值默认为 None (或者 np.nan)
            entry = {k: None for k in headers}
            entry['Domain'] = domain_name
            entry['Label'] = domain_label_map[domain_name]
            all_samples[domain_name] = entry

        return all_samples[domain_name]

    # ================= 3. 处理 QoS 和 Runtime 数据 =================
    print("Processing QoS & Runtime data...")
    for i in tqdm(range(1, 11), desc="QoS/Runtime Batch"):
        # 路径逻辑保持不变
        if i in [8, 10]:
            base_path = f'../qos/out_batch_{i}/out_batch_mp/'
        else:
            base_path = f'../qos/out_batch_{i}/out_batch/'

        if not os.path.exists(base_path):
            continue

        domain_path_list = os.listdir(base_path)

        for d in domain_path_list:
            entry = get_sample_entry(d)
            if entry is None: continue  # 不是目标域名，跳过

            # --- QoS ---
            round_folder = 'round_1'  # 假设总是 round_1
            qos_file = os.path.join(base_path, d, round_folder, 'qos.json')

            flag = 1
            if os.path.isfile(qos_file):
                pass
            elif os.path.isfile(os.path.join(base_path, d, 'round_2', 'qos.json')):
                qos_file = os.path.join(base_path, d, 'round_2', 'qos.json')
            elif os.path.isfile(os.path.join(base_path, d, 'round_3', 'qos.json')):
                qos_file = os.path.join(base_path, d, 'round_3', 'qos.json')
            elif os.path.isfile(os.path.join(base_path, d, 'round_4', 'qos.json')):
                qos_file = os.path.join(base_path, d, 'round_4', 'qos.json')
            else:
                flag = 0
            if flag:
                try:
                    with open(qos_file, 'r', encoding='utf-8') as f:
                        qd = json.load(f)  # qos data

                        entry['DNS Lookup Time'] = qd.get('dns_ms')
                        entry['TCP Handshake'] = qd.get('tcp_ms')
                        entry['TLS Setup'] = qd.get('tls_ms')
                        entry['TTFB'] = qd.get('ttfb_ms')
                        entry['Load Time'] = qd.get('load_ms')
                        entry['Request Count'] = qd.get('request_count')
                        entry['Total Transferred Bytes'] = qd.get('total_transferred_bytes')
                        entry['Avg Resource Bytes'] = qd.get('avg_resource_bytes')
                        entry['Connection Reuse Ratio'] = qd.get('connection_reuse_ratio')
                        entry['CDN Ratio'] = qd.get('cdn_evidence_ratio')

                        # 计算 Ratio
                        h = qd.get('http_protocol_dist', {})
                        h_total = sum(h.values())
                        entry['HTTP3 Ratio'] = h.get('h3', 0) / h_total if h_total > 0 else 0

                        t = qd.get('tls_protocol_dist', {})
                        t_total = sum(t.values())
                        entry['TLS 1.3 Ratio'] = t.get('TLS 1.3', 0) / t_total if t_total > 0 else 0

                except Exception as e:
                    # print(f"Error reading QoS for {d}: {e}")
                    pass

            # --- Runtime ---
            trace_file = os.path.join(base_path, d, round_folder, 'trace.json')
            if os.path.isfile(trace_file):
                try:
                    with open(trace_file, 'r', encoding='utf-8') as f:
                        rd = json.load(f)  # runtime data
                        entry['JS Exec Time'] = rd.get('js_exec_time_ms')
                        entry['JS Long Task Count'] = rd.get('js_exec_event_count')
                        entry['Layout Count'] = rd.get('layout_count')
                        entry['Paint Count'] = rd.get('paint_count')
                        entry['Style Recalc'] = rd.get('style_recalc_count')
                except Exception as e:
                    pass

    # ================= 4. 处理 QoE 数据 =================
    print("Processing QoE data...")
    for i in tqdm(range(1, 11), desc="QoE Batch"):
        if i in [4, 5, 6, 8]:
            base_path = f'../qoe/out_batch_qoe_{i}/out_batch_qoe_{i}/'
        elif i == 10:
            base_path = f'../qoe/out_batch_qoe_{i}/'
        else:
            base_path = f'../qoe/out_batch_qoe_{i}/out_batch_qoe/'

        if not os.path.exists(base_path):
            continue

        domain_path_list = os.listdir(base_path)

        for d in domain_path_list:
            entry = get_sample_entry(d)
            if entry is None: continue

            # QoE 文件路径
            if i in [4, 5, 6, 8, 10]:
                lh_file = os.path.join(base_path, d, 'report.json')
            else:
                lh_file = os.path.join(base_path, d, 'round_001/lh.json')

            if os.path.isfile(lh_file):
                try:
                    with open(lh_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        audits = data.get('audits', {})

                        # 辅助取值函数，防止 None 报错
                        def get_audit_val(key):
                            val = audits.get(key, {}).get('numericValue', None)
                            return val if val is not None else None

                        entry['LCP'] = get_audit_val('largest-contentful-paint')
                        entry['FCP'] = get_audit_val('first-contentful-paint')
                        entry['CLS'] = get_audit_val('cumulative-layout-shift')
                        entry['TTI'] = get_audit_val('interactive')
                        entry['TBT'] = get_audit_val('total-blocking-time')
                        entry['Speed Index'] = get_audit_val('speed-index')

                except Exception as e:
                    print(f"Error reading QoE for {d}: {e}")

    # ================= 5. 保存结果 =================
    output_file = '../preprocessed_data/domain_metrics_full.csv'
    print(f"Saving results to {output_file}...")

    # 统计一下有多少样本
    print(f"Total valid samples collected: {len(all_samples)}")

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        # 遍历所有收集到的样本
        for d, row_data in all_samples.items():
            writer.writerow(row_data)

    print("Done!")


if __name__ == '__main__':
    main_result()
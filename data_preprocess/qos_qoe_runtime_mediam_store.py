import copy
import csv
import os
import json
from tqdm import trange
import numpy as np

def main_result():
    with open('../domain_fine_qoe_pkt.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        domain_list = [row[0] for row in reader]

    # 需要先将domain列表分成三份
    with open('../url_list_1800.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        domain_1800 = [row[0].split('//')[1].replace('.', '_') for row in reader]

    domain_1800_1 = domain_1800[:600]
    domain_1800_2 = domain_1800[600:1200]
    domain_1800_3 = domain_1800[1200:]

    domain_top, domain_torso, domain_tail = [], [], []

    for domain in domain_list:
        if domain in domain_1800_1:
            domain_top.append(domain)
        elif domain in domain_1800_2:
            domain_torso.append(domain)
        elif domain in domain_1800_3:
            domain_tail.append(domain)
        else:
            print('error')


    qos_result_dict_top = {
        'DNS Lookup Time': [],
        'TCP Handshake': [],
        'TLS Setup': [],
        'TTFB': [],
        'Load Time': [],
        'Request Count': [],
        'Total Transferred Bytes': [],
        'Avg Resource Bytes': [],
        'HTTP3 Ratio': [],
        'TLS 1.3 Ratio': [],
        'Connection Reuse Ratio': [],
        'CDN Ratio': []
    }
    qos_result_dict_torso = copy.deepcopy(qos_result_dict_top)
    qos_result_dict_tail = copy.deepcopy(qos_result_dict_top)

    runtime_result_dict_top = {
        'JS Exec Time': [],
        'JS Long Task Count': [],
        'Layout Count': [],
        'Paint Count': [],
        'Style Recalc': []
    }
    runtime_result_dict_torso = copy.deepcopy(runtime_result_dict_top)
    runtime_result_dict_tail = copy.deepcopy(runtime_result_dict_top)

    qoe_result_dict_top = {
        'LCP': [],
        'FCP': [],
        'CLS': [],
        'TTI': [],
        'TBT': [],
        'Speed Index': []
    }
    qoe_result_dict_torso = copy.deepcopy(qoe_result_dict_top)
    qoe_result_dict_tail = copy.deepcopy(qoe_result_dict_top)





    # 处理qos和runtime数据
    for i in trange(1, 11):
        if i in [8, 10]:
            domain_path = os.listdir(f'../qos/out_batch_{i}/out_batch_mp/')
        else:
            domain_path = os.listdir(f'../qos/out_batch_{i}/out_batch/')
        for d in domain_path:
            # ----------------处理qos指标------------------------
            if i in [8, 10]:
                file = f'../qos/out_batch_{i}/out_batch_mp/{d}/round_1/qos.json'
            else:
                file = f'../qos/out_batch_{i}/out_batch/{d}/round_1/qos.json'
            if os.path.isfile(file):
                with open(file, 'r', encoding='utf-8') as f:
                    qos_data = json.load(f)
                    target_dict = None
                    if d in domain_top:
                        target_dict = qos_result_dict_top
                    elif d in domain_torso:
                        target_dict = qos_result_dict_torso
                    elif d in domain_tail:
                        target_dict = qos_result_dict_tail

                    if target_dict is not None:
                        try:
                            target_dict['DNS Lookup Time'].append(qos_data['dns_ms'])
                            target_dict['TCP Handshake'].append(qos_data['tcp_ms'])
                            target_dict['TLS Setup'].append(qos_data['tls_ms'])
                            target_dict['TTFB'].append(qos_data['ttfb_ms'])
                            target_dict['Load Time'].append(qos_data['load_ms'])
                            target_dict['Request Count'].append(qos_data['request_count'])
                            target_dict['Total Transferred Bytes'].append(qos_data['total_transferred_bytes'])
                            target_dict['Avg Resource Bytes'].append(qos_data['avg_resource_bytes'])
                            h = qos_data['http_protocol_dist']
                            target_dict['HTTP3 Ratio'].append(h.get('h3', 0) / sum(h.values()) if d else 0)
                            t = qos_data['tls_protocol_dist']
                            target_dict['TLS 1.3 Ratio'].append(t.get('TLS 1.3', 0) / sum(t.values()) if d else 0)
                            target_dict['Connection Reuse Ratio'].append(qos_data['connection_reuse_ratio'])
                            target_dict['CDN Ratio'].append(qos_data['cdn_evidence_ratio'])
                        except KeyError:
                            pass
                            # print(i, d, KeyError)
            # ----------------处理runtime指标------------------------
            if i in [8, 10]:
                file = f'../qos/out_batch_{i}/out_batch_mp/{d}/round_1/trace.json'
            else:
                file = f'../qos/out_batch_{i}/out_batch/{d}/round_1/trace.json'
            if os.path.isfile(file):
                with open(file, 'r', encoding='utf-8') as f:
                    runtime_data = json.load(f)
                    target_dict = None
                    if d in domain_top:
                        target_dict = runtime_result_dict_top
                    elif d in domain_torso:
                        target_dict = runtime_result_dict_torso
                    elif d in domain_tail:
                        target_dict = runtime_result_dict_tail

                    if target_dict is not None:
                        try:
                            target_dict['JS Exec Time'].append(runtime_data['js_exec_time_ms'])
                            target_dict['JS Long Task Count'].append(runtime_data['js_exec_event_count'])
                            target_dict['Layout Count'].append(runtime_data['layout_count'])
                            target_dict['Paint Count'].append(runtime_data['paint_count'])
                            target_dict['Style Recalc'].append(runtime_data['style_recalc_count'])

                        except KeyError:
                            pass

    # 处理qoe数据
    for i in trange(1, 11):
        if i in [4, 5, 6, 8, 10]:
            domain_path = os.listdir(f'../qoe/out_batch_qoe_{i}/out_batch_qoe_{i}/')
        else:
            domain_path = os.listdir(f'../qoe/out_batch_qoe_{i}/out_batch_qoe/')
        for d in domain_path:
            if i in [4, 5, 6, 8, 10]:
                file = f'../qoe/out_batch_qoe_{i}/out_batch_qoe_{i}/{d}/round_001/lh.json'
            else:
                file = f'../qoe/out_batch_qoe_{i}/out_batch_qoe/{d}/round_001/lh.json'

            target_dict = None
            if d in domain_top:
                target_dict = qoe_result_dict_top
            elif d in domain_torso:
                target_dict = qoe_result_dict_torso
            elif d in domain_tail:
                target_dict = qoe_result_dict_tail
            if target_dict:
                if not os.path.exists(file):
                    print(f"错误: 找不到文件 {file}")
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"读取 JSON 失败: {e}")
                audits = data.get('audits', {})
                target_dict['LCP'].append(audits.get('largest-contentful-paint').get('numericValue', 'N/A'))
                target_dict['FCP'].append(audits.get('first-contentful-paint').get('numericValue', 'N/A'))
                target_dict['CLS'].append(audits.get('cumulative-layout-shift').get('numericValue', 'N/A'))
                target_dict['TTI'].append(audits.get('interactive').get('numericValue', 'N/A'))
                target_dict['TBT'].append(audits.get('total-blocking-time').get('numericValue', 'N/A'))
                target_dict['Speed Index'].append(audits.get('speed-index').get('numericValue', 'N/A'))



    processed_data_top = {
        k: (np.mean(v) if k == 'HTTP3 Ratio' else np.median(v))
        for k, v in qos_result_dict_top.items() if v
    }

    processed_data_torso = {
        k: (np.mean(v) if k == 'HTTP3 Ratio' else np.median(v))
        for k, v in qos_result_dict_torso.items() if v
    }

    processed_data_tail = {
        k: (np.mean(v) if k == 'HTTP3 Ratio' else np.median(v))
        for k, v in qos_result_dict_tail.items() if v
    }

    runtime_processed_top = {k: np.median(v) for k, v in runtime_result_dict_top.items() if v}
    runtime_processed_torso = {k: np.median(v) for k, v in runtime_result_dict_torso.items() if v}
    runtime_processed_tail = {k: np.median(v) for k, v in runtime_result_dict_tail.items() if v}

    # print(qoe_result_dict_top)
    qoe_processed_top = {k: np.median([x for x in v if not isinstance(x, str)]) for k, v in qoe_result_dict_top.items() if v}
    qoe_processed_torso = {k: np.median([x for x in v if not isinstance(x, str)]) for k, v in qoe_result_dict_torso.items() if v}
    qoe_processed_tail = {k: np.median([x for x in v if not isinstance(x, str)]) for k, v in qoe_result_dict_tail.items() if v}

    return processed_data_top, processed_data_torso, processed_data_tail, runtime_processed_top, runtime_processed_torso, runtime_processed_tail, qoe_processed_top, qoe_processed_torso, qoe_processed_tail


if __name__ == '__main__':
    result = main_result()
    for item in result:
        print(item)
import csv
import os
import json
import numpy as np

# 将派生指标保存成csv格式

file = '../preprocessed_data/domain_metrics_full.csv'

with open(file, 'r') as f:
    reader = csv.reader(f)
    data = [row for row in reader]

data = data[1:]

result_dict = {}

for sample in data:
    label = sample[0]
    try:
        LCP_TotalByte = float(sample[-6]) / float(sample[8])
        LCP_RequestCount = float(sample[-6]) / float(sample[7])
        TTFB_LCP = float(sample[5]) / float(sample[-6])
    except:
        LCP_TotalByte = 0
        LCP_RequestCount = 0
        TTFB_LCP = 0

    result_dict[label] = [LCP_TotalByte, LCP_RequestCount, TTFB_LCP]

# 保存全部的键
labels = set(list(result_dict.keys()))
for i in range(1, 11):
    if i in [8, 10]:
        pass
    else:
        path_list = os.listdir(f'../qos/out_batch_{i}/out_batch/')
        for path in path_list:
            if path in labels:
                curr_file = f'../qos/out_batch_{i}/out_batch/{path}/round_1/derived_metrics.json'
                try:
                    with open(curr_file, 'r') as f:
                        curr_data = json.load(f)
                        result_dict[path].extend([curr_data['resource_entropy'], curr_data['request_size_entropy'], curr_data['request_timing_entropy']])
                        # print('ok')
                except FileNotFoundError:
                    pass

# 计算qos类的变异系数
for i in range(1, 11):
    if i in [8, 10]:
        pass
    else:
        path_list = os.listdir(f'../qos/out_batch_{i}/out_batch/')
        for path in path_list:
            if path in labels:
                try:
                    qos_4time_http3 = []
                    qos_4time_tcp = []
                    qos_4time_load = []
                    for j in range(1, 5):
                        curr_file = f'../qos/out_batch_{i}/out_batch/{path}/round_{j}/qos.json'
                        with open(curr_file, 'r') as f:
                            curr_data = json.load(f)
                            h = curr_data["http_protocol_dist"]
                            qos_4time_http3.append(h.get('h3', 0) / sum(h.values()))
                            qos_4time_tcp.append(curr_data["tcp_ms"])
                            qos_4time_load.append(curr_data["load_ms"])
                    result_dict[path].append(np.std(qos_4time_http3) / np.mean(qos_4time_http3))
                    result_dict[path].append(np.std(qos_4time_tcp) / np.mean(qos_4time_tcp))
                    result_dict[path].append(np.std(qos_4time_load) / np.mean(qos_4time_load))
                except FileNotFoundError:
                    pass


# 计算runtime类的变异系数
for i in range(1, 11):
    if i in [8, 10]:
        pass
    else:
        path_list = os.listdir(f'../qos/out_batch_{i}/out_batch/')
        for path in path_list:
            if path in labels:
                try:
                    runtime_4time_js = []
                    runtime_4time_paint = []
                    runtime_4time_style = []
                    for j in range(1, 5):
                        curr_file = f'../qos/out_batch_{i}/out_batch/{path}/round_{j}/trace.json'
                        with open(curr_file, 'r') as f:
                            curr_data = json.load(f)
                            runtime_4time_js.append(curr_data["js_exec_time_ms"])
                            runtime_4time_paint.append(curr_data["paint_count"])
                            runtime_4time_style.append(curr_data["style_recalc_count"])
                    result_dict[path].append(np.std(runtime_4time_js) / np.mean(runtime_4time_js))
                    result_dict[path].append(np.std(runtime_4time_paint) / np.mean(runtime_4time_paint))
                    result_dict[path].append(np.std(runtime_4time_style) / np.mean(runtime_4time_style))
                except FileNotFoundError:
                    pass

output = [['label', 'LCP_TotalByte', 'LCP_RequestCount', 'TTFB_LCP', 'resource_entropy', 'request_size_entropy', 'request_timing_entropy', 'http_protocol_cv', 'tcp_cv', 'load_cv', 'js_cv', 'paint_cv', 'style_cv']]

for k, v in result_dict.items():
    output.append([k] + v)

output_new = []
for item in output:
    if len(item) > 11:
        output_new.append(item)

with open('../preprocessed_data/domain_derived_cv_metrics.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(output_new)
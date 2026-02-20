import os
import numpy as np
from scapy.all import rdpcap
from sklearn.preprocessing import LabelEncoder
import pickle
from tqdm import trange, tqdm
import multiprocessing as mp
import logging
import csv
import glob

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_pcap(file_path):
    packets = rdpcap(file_path)
    direction_list = []

    for packet in packets:
        if packet.haslayer('IP'):
            src_ip = packet['IP'].src
            if src_ip.startswith('172.31.'):
                direction_list.append(len(packet))
            else:
                direction_list.append(-len(packet))

    # If the list length exceeds 5000, truncate to the first 5000
    if len(direction_list) > 5000:
        direction_list = direction_list[:5000]
    # If the list length is less than 5000, pad with zeros
    else:
        direction_list.extend([0] * (5000 - len(direction_list)))

    return direction_list

def process_file(file_path):
    try:
        trace = process_pcap(file_path)
        label = file_path.split('/')[-2]
        return trace, label
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return None, None

def main_process():
    trace_all = []
    label_all = []
    file_paths = []

    with open('../iwqos_project/domain_fine_qoe_pkt.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        domain_list = [row[0] for row in reader]

    for i in range(1, 11):
        path_list = os.listdir(f'../traffic/out_pcap_{i}/out_pcap/')
        for d in path_list:
            if d in domain_list:
                pcap_files_curr = glob.glob(os.path.join(f'../traffic/out_pcap_{i}/out_pcap/{d}', "*.pcap"))
                file_paths.extend(pcap_files_curr)

    with mp.Pool(processes=60) as pool:
        # 使用 imap_unordered + tqdm 显示进度
        results = []
        for result in tqdm(pool.imap_unordered(process_file, file_paths), total=len(file_paths)):
            trace, label = result
            if trace is not None and label is not None:
                trace_all.append(trace)
                label_all.append(label)
                # logging.info(f'Processed file with label {label}')

    traces = np.array(trace_all)
    le = LabelEncoder()
    numeric_labels = np.array(le.fit_transform(label_all))

    logging.info(f'Traces shape: {traces.shape}')
    logging.info(f'Labels shape: {numeric_labels.shape}')

    with open('../preprocessed_data/DF_X_with_size.pkl', 'wb') as f:
        pickle.dump(traces, f)
    with open('../preprocessed_data/DF_y_with_size.pkl', 'wb') as f:
        pickle.dump(numeric_labels, f)

if __name__ == '__main__':
    main_process()

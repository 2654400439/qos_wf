from scapy.all import rdpcap, wrpcap, IP, IPv6, TCP
from collections import Counter
import os
from tqdm import trange
import random
import csv
import glob

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

file_list = random.sample(file_list, 200)

print(len(file_list))

s1_list = []
ek_list = []
K_list = []

for i in trange(len(file_list)):
    pcap_file = file_list[i]
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


    sequence = process_pcap(pcap_file)
    # print(sequence)

    K = 0
    flag = 0
    for i in range(len(sequence)):
        if sequence[i] < 0:
            flag += 1
        else:
            if flag >= 4:
                K += 1
            else:
                pass
            flag = 0
    K_list.append(K)

print(Counter(K_list))

# print(sum(s1_list) / len(s1_list))
# print(sum(ek_list) / len(ek_list))

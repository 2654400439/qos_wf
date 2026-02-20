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


# for i in range(3, 11):
#     for j in range(1, 41):
#         filefolder = f"/data/disk1/cyfsec/H123/result_{i}/result_{i}_{j}/result_{i}_{j}/pcap"
#         files = os.listdir(filefolder)
#         file_list = [filefolder + '/' + item for item in files]
file_list = random.sample(file_list, 200)
# file_list = random.sample(file_list, 200)

# print(len(file_list))

s1_list = []
ek_list = []

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

    U0 = [item if item > 0 else 0 for item in sequence]
    for i in range(len(U0) - 1):
        U0[i+1] = U0[i+1] + U0[i]

    counter = dict(Counter(U0))

    A = {key: value for key, value in counter.items() if value >= 4}
    # print(A)

    S = {key: U0.index(key) for key in A.keys()}
    # print(S)

    E = {key: len(U0) - 1 - U0[::-1].index(key) for key in A.keys()}
    # print(E)


    si = list(S.values())
    ei = list(E.values())
    ui = list(S.keys())

    B = [(si[i], ei[i], ui[i]) for i in range(len(si))]

    s1 = 0
    ek = 0
    for item in si:
        if item > 20:
            s1 = item
            break

    tmp = 0
    for item in ei:
        if item > 100:
            ek = tmp
            break
        tmp = item

    s1_list.append(s1)
    ek_list.append(ek)
    # print(s1, ek)

print(sum(s1_list) / len(s1_list))
print(sum(ek_list) / len(ek_list))

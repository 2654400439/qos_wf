from scapy.all import rdpcap
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import csv
import glob


def pcap_to_txt(args):
    input_file, output_file, max_packets = args

    try:
        packets = rdpcap(input_file)
        if len(packets) == 0:
            return

        start_time = packets[0].time

        with open(output_file, 'w') as f:
            for packet in packets[:max_packets]:
                relative_time = packet.time - start_time
                packet_size = len(packet)
                f.write(f"{relative_time}\t{packet_size}\n")
    except Exception as e:
        print(f"[ERROR] Failed to process {input_file}: {e}")


def prepare_file_list():
    # tasks = []
    # for i in range(3, 11):
    #     for j in range(1, 41):
    #         filefolder = f"/data/disk1/cyfsec/H123/result_{i}/result_{i}_{j}/result_{i}_{j}/pcap/"
    #         if not os.path.exists(filefolder):
    #             continue
    #         for file in os.listdir(filefolder):
    #             input_file = os.path.join(filefolder, file)
    #             index = str(j)
    #             output_file = f"/data/disk1/cyfsec/STAR/RF/{file.split('.')[0]}-{index}.txt"
    #             tasks.append((input_file, output_file, 5000))

    tasks = []

    with open('../iwqos_project/domain_fine_qoe_pkt.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        domain_list = [row[0] for row in reader]

    for i in range(1, 11):
        path_list = os.listdir(f'../traffic/out_pcap_{i}/out_pcap/')
        for d in path_list:
            if d in domain_list:
                pcap_files_curr = glob.glob(os.path.join(f'../traffic/out_pcap_{i}/out_pcap/{d}', "*.pcap"))
                for p in pcap_files_curr:
                    input_file = p
                    index = str(p.split('/')[-1].split('_')[-1].split('.')[0])
                    output_file = f"../preprocessed_data/RF/{d}-{index}.txt"
                    tasks.append((input_file, output_file, 5000))
    return tasks


if __name__ == '__main__':
    tasks = prepare_file_list()
    print(f"Total files to process: {len(tasks)}")

    with Pool(processes=min(cpu_count(), 32)) as pool:  # 可以根据你的服务器调整进程数
        list(tqdm(pool.imap_unordered(pcap_to_txt, tasks), total=len(tasks)))

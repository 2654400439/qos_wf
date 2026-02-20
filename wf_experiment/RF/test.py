import numpy as np
import torch
from RF import getRF
import torch.utils.data as Data
import const_rf as const
import csv
import pre_recall
import os
import json
from sklearn.metrics import classification_report, precision_recall_fscore_support


def load_data(fpath):
    data = np.load(fpath, allow_pickle=True).item()
    train_X, train_y = data['dataset'], data['label']
    return train_X, train_y


def load_model(class_num, path, device):
    model = getRF(class_num)
    # 确保路径拼接正确
    model.load_state_dict(torch.load(os.path.join(path, 'Undefended.pth'), map_location=device))
    model = model.to(device)
    return model


if __name__ == '__main__':
    # 1. 基础配置
    test_dataset = ['../preprocessed_data/RF_features-test.npy']
    mapping_file = '../preprocessed_data/RF_category_mapping.json'
    output_npz_file = '../preprocessed_data/rf_class_f1_scores.npz'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    defense_model = load_model(const.num_classes, 'pretrained/', device).eval()

    # 2. 加载类别映射关系
    print(f"Loading category mapping from {mapping_file}...")
    with open(mapping_file, "r") as f:
        category_map = json.load(f)  # 格式: {"google": 1, "youtube": 2, ...}

    # 打印部分映射以确认是否是从1开始
    first_few_items = list(category_map.items())[:3]
    print(f"Sample mapping: {first_few_items}")

    for path in test_dataset:
        print(f"Testing on {path}...")
        features, test_y = load_data(path)

        test_x = torch.unsqueeze(torch.from_numpy(features), dim=1).type(torch.FloatTensor).to(device)
        test_y = torch.squeeze(torch.from_numpy(test_y)).type(torch.LongTensor)

        # 【核心步骤 A】：标签偏移处理 (Raw -> Model)
        # 假设原始标签是 1-based (1..1173)，模型需要 0-based (0..1172)
        if test_y.min() >= 1:
            print("Detected 1-based labels in test set. Shifting labels by -1...")
            test_y = test_y - 1

        test_data = Data.TensorDataset(test_x, test_y)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)

        top1_correct = 0
        top5_correct = 0
        total = 0
        website_res = []

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                output = defense_model(x).cpu().squeeze().numpy()
                label = y.item()  # 这是已经 -1 后的 0-based 标签
                total += 1

                top1 = int(np.argmax(output))
                if output.ndim == 0:
                    top5 = [top1]
                else:
                    top5 = np.argsort(output)[-5:][::-1].tolist()

                if top1 == label:
                    top1_correct += 1
                if label in top5:
                    top5_correct += 1

                website_res.append([label, top1, top5])

                all_preds.append(top1)
                all_labels.append(label)

        # --- CSV 保存逻辑 (保持不变) ---
        os.makedirs('result', exist_ok=True)
        with open('result/1.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            for label, top1, top5 in website_res:
                writer.writerow([label, top1] + top5)

        with open('result/top1_only.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            for label, top1, _ in website_res:
                writer.writerow([label, top1])

        # --- 打印基础指标 ---
        print("-" * 30)
        print(f"Total samples: {total}")
        print(f"Top-1 Accuracy: {top1_correct / total:.4f}")
        print(f"Top-5 Accuracy: {top5_correct / total:.4f}")

        # --- 【核心步骤 B】：计算每个类别的 F1 ---
        print("-" * 30)
        print("Calculating Per-Class Metrics...")

        # 强制指定 labels 参数为 0 到 num_classes-1
        # 这样返回的 f1_scores 数组的索引 i 就严格对应模型类别 i
        # zero_division=0 防止除以零报错
        target_labels = list(range(const.num_classes))
        precision, recall, f1_scores, support = precision_recall_fscore_support(
            all_labels,
            all_preds,
            labels=target_labels,
            average=None,
            zero_division=0
        )

        # --- 【核心步骤 C】：映射还原并保存 ---
        # 这里的逻辑是：
        # category_map: {"google": 1}  (Raw ID)
        # f1_scores: [0.95, ...]       (Model ID 0 的分数)
        # 关系: Model ID = Raw ID - 1

        class_f1_mapping = {}
        for label_str, raw_id in category_map.items():
            # 还原逻辑：再次处理标签偏移
            model_id = raw_id - 1

            # 确保不越界 (防止 json 里的类别比模型定义的还多)
            if 0 <= model_id < len(f1_scores):
                class_f1_mapping[label_str] = f1_scores[model_id]
            else:
                # 这种情况理论不该发生，除非 const.num_classes 设小了
                print(f"Warning: ID {raw_id} (Model ID {model_id}) out of bounds for F1 scores array.")

        print(f"Constructed mapping with {len(class_f1_mapping)} classes.")

        # 保存为 .npz
        print(f"Saving per-class F1 scores to {output_npz_file}...")
        np.savez(output_npz_file, **class_f1_mapping)
        print("Done.")
        print("-" * 30)
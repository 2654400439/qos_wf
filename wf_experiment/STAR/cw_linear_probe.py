import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from sklearn.metrics import f1_score
from collections import defaultdict
from traffic_encoder_3d import DFEncoder

# ================= 配置 =================
CACHE_PATH = '../preprocessed_data/star_traffic_dataset.npz'
CHECKPOINT_PATH = 'STAR_model_pt/best_STAR_model.pt'
OUTPUT_SAVE_PATH = '../preprocessed_data/star_class_f1_scores.npz'

NUM_RUNS = 5  # 重复训练次数
BATCH_SIZE = 128
EPOCHS = 40  # 每次训练的轮数
NUM_SUPPORT = 10  # K-shot 样本数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def remove_module_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


# ================= 1. 数据与模型加载 (只做一次) =================
print("[INFO] Initializing environment...")

# 加载数据
if os.path.exists(CACHE_PATH):
    print(f"[INFO] Loading cached dataset from {CACHE_PATH}")
    cache = np.load(CACHE_PATH, allow_pickle=True)

    # 原始数据
    raw_traffic = torch.tensor(cache['traffic'], dtype=torch.float32)
    labels = torch.tensor(cache['label'], dtype=torch.long)

    # 恢复标签映射: {0: 'google.com', ...}
    raw_mapping = cache['label_mapping'].item()
    idx_to_domain = {v: k for k, v in raw_mapping.items()}
else:
    raise FileNotFoundError(f"Dataset not found at {CACHE_PATH}")

# 加载模型
print(f"[INFO] Loading checkpoint from {CHECKPOINT_PATH}")
traffic_encoder = DFEncoder(input_length=5000).to(device)
projection_traffic = nn.Linear(512, 256).to(device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
traffic_encoder.load_state_dict(remove_module_prefix(checkpoint['traffic_encoder']))
projection_traffic.load_state_dict(remove_module_prefix(checkpoint['projection_traffic']))

traffic_encoder.eval()
projection_traffic.eval()

# ================= 2. 特征预提取 (性能优化) =================
# 因为 Encoder 是冻结的，我们先把所有数据的特征算出来，这样后面跑5次训练会非常快
print("[INFO] Pre-computing features to speed up multi-run training...")
all_features = []
data_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(raw_traffic),
    batch_size=BATCH_SIZE,
    shuffle=False
)

with torch.no_grad():
    for (batch_x,) in data_loader:
        batch_x = batch_x.to(device)
        embeds = traffic_encoder(batch_x)
        embeds = projection_traffic(embeds)
        all_features.append(embeds.cpu())  # 先放回 CPU 省显存

# 替换原始的 traffic_vectors，现在它是 (N, 256) 的特征矩阵了
traffic_vectors = torch.cat(all_features)
print(f"[INFO] Features pre-computed. Shape: {traffic_vectors.shape}")

# ================= 3. 多轮训练主循环 =================
# 用于存储每个类别在 5 次运行中的所有 F1 分数
# 结构: {'google.com': [0.8, 0.82, ...], 'youtube.com': [...]}
f1_accumulator = defaultdict(list)

for run in range(NUM_RUNS):
    print(f"\n{'=' * 20} Run {run + 1}/{NUM_RUNS} {'=' * 20}")

    # 设置不同的随机种子以保证每次的数据划分不同
    current_seed = 42 + run
    torch.manual_seed(current_seed)
    np.random.seed(current_seed)

    # --- 数据划分 (K-shot Split) ---
    labels_np = labels.numpy()
    train_idx, val_idx = [], []

    # 动态筛选有效类别
    valid_classes_raw = []
    for cls in np.unique(labels_np):
        idx_cls = np.where(labels_np == cls)[0]
        if len(idx_cls) < NUM_SUPPORT + 1:
            continue
        valid_classes_raw.append(cls)
        # 随机打乱
        np.random.shuffle(idx_cls)
        train_idx.extend(idx_cls[:NUM_SUPPORT])
        val_idx.extend(idx_cls[NUM_SUPPORT:])

    train_data = traffic_vectors[train_idx]
    train_targets_raw = labels[train_idx]
    val_data = traffic_vectors[val_idx]
    val_targets_raw = labels[val_idx]

    # --- 标签重映射 (Remap) ---
    # 找出当前训练集里存在的唯一类别
    unique_classes = torch.unique(train_targets_raw)
    sorted_unique = torch.sort(unique_classes)[0]

    # Old ID -> New ID (0..N)
    class_map = {old.item(): new for new, old in enumerate(sorted_unique)}
    # New ID -> Old ID (为了最后还原)
    new_to_old_map = {new: old.item() for new, old in enumerate(sorted_unique)}

    actual_num_classes = len(unique_classes)


    def remap(lab_tensor, mapping):
        lab_np = lab_tensor.numpy()
        return torch.tensor([mapping[l] for l in lab_np], dtype=torch.long)


    train_y = remap(train_targets_raw, class_map)
    val_y = remap(val_targets_raw, class_map)

    # --- 训练分类器 ---
    from torch.utils.data import TensorDataset, DataLoader

    train_loader = DataLoader(TensorDataset(train_data, train_y), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data, val_y), batch_size=BATCH_SIZE, shuffle=False)

    # 每次都要重新初始化一个新的分类器
    classifier = nn.Linear(256, actual_num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        classifier.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = classifier(bx)  # 输入已经是特征了，直接进分类器
            loss = F.cross_entropy(logits, by)
            loss.backward()
            optimizer.step()

    # --- 验证并记录 F1 ---
    classifier.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            logits = classifier(bx)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(by.cpu().numpy())

    # 计算本次运行每个类别的 F1
    run_f1_scores = f1_score(all_targets, all_preds, average=None)

    # --- 将 F1 分数映射回域名并存储 ---
    # run_f1_scores 的索引是 New ID (0..N)
    for new_id, score in enumerate(run_f1_scores):
        old_id = new_to_old_map[new_id]  # 找回原始数字标签
        domain_name = idx_to_domain[old_id]  # 找回域名字符串
        f1_accumulator[domain_name].append(score)

    print(f"Run {run + 1} finished. Macro F1: {np.mean(run_f1_scores):.4f}")

# ================= 4. 计算平均值并保存 =================
print("\n" + "=" * 40)
print(f"{'Domain Name':<30} | {'Avg F1 Score':<10}")
print("-" * 40)

final_avg_f1_map = {}

# 计算平均值
for domain, scores in f1_accumulator.items():
    avg_score = np.mean(scores)
    final_avg_f1_map[domain] = avg_score
    print(f"{domain:<30} | {avg_score:.4f}")

# 保存为 .npz 文件
# 这里的 keys 是域名(str)，values 是 float
# np.savez 会把 key 当作数组名，value 当作数组内容保存
print(f"\n[INFO] Saving averaged results to {OUTPUT_SAVE_PATH}")

# 确保保存目录存在
os.makedirs(os.path.dirname(OUTPUT_SAVE_PATH), exist_ok=True)

np.savez(OUTPUT_SAVE_PATH, **final_avg_f1_map)

print("All done!")
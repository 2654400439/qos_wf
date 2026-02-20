import os
import sys
import csv
import glob
import pickle
import numpy as np
import random
from collections import Counter
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
# 假设 utility 和 Model_NoDef 在同一目录下
from utility import LoadDataNoDefCW
from Model_NoDef import DFNet

# =========================================================
# 0. 环境与硬件配置 (保持不变)
# =========================================================

# --- CUDA 编译器路径修复 ---
try:
    import nvidia.cuda_nvcc
    import nvidia.cudnn

    nvcc_path = os.path.dirname(nvidia.cuda_nvcc.__file__)
    cudnn_path = os.path.dirname(nvidia.cudnn.__file__)
    os.environ['PATH'] = os.path.join(nvcc_path, 'bin') + os.pathsep + os.environ.get('PATH', '')
    os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={nvcc_path}"
    print(f"✅ 成功配置 CUDA 编译器路径: {nvcc_path}")
except ImportError:
    print("❌ 未找到 nvidia-cuda-nvcc，请确保运行了 pip install...")

# --- TF 配置 ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
description = "Training and evaluating DF model for closed-world scenario (Average of 5 Runs)"
print(description)


def safe_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


# =========================================================
# 1. 预备工作：构建标签映射 (只运行一次，避免重复IO)
# =========================================================
print("\n[Step 1/5] 正在构建字符串标签映射...")

# 1.1 加载白名单
domain_list = []
try:
    with open('../iwqos_project/domain_fine_qoe_pkt.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        domain_list = [row[0] for row in reader]
except Exception as e:
    print(f"❌ 无法读取 CSV: {e}")
    exit(1)

# 1.2 扫描文件夹重建原始 Label 集合
found_labels_set = set()
for i in range(1, 11):
    base_path = f'../traffic/out_pcap_{i}/out_pcap/'
    if not os.path.exists(base_path): continue
    for d in os.listdir(base_path):
        if d in domain_list:
            # 只要里面有文件就算有效类别
            if glob.glob(os.path.join(base_path, d, "*.pcap")):
                found_labels_set.add(d)

# 1.3 原始字符串 -> 原始 ID 的映射 (基于字典序)
all_labels_sorted = sorted(list(found_labels_set))
original_id_to_string = {idx: label for idx, label in enumerate(all_labels_sorted)}
print(f"   扫描完成。共找到 {len(original_id_to_string)} 个原始类别。")

# 1.4 复现训练集筛选逻辑 (Count >= 10)
print("   复现筛选逻辑...")
with open('../preprocessed_data/DF_y_with_size.pkl', 'rb') as f:
    y_raw_ids = pickle.load(f)
class_counts = Counter(y_raw_ids)
eligible_original_ids = [oid for oid, count in class_counts.items() if count >= 10]
sorted_eligible_ids = sorted(eligible_original_ids)

# 1.5 最终映射：模型输出 ID (0~1172) -> 字符串 Label
train_id_to_string_map = {}
for train_id, original_id in enumerate(sorted_eligible_ids):
    if original_id in original_id_to_string:
        train_id_to_string_map[train_id] = original_id_to_string[original_id]

print(f"   映射构建完成。模型有效分类数: {len(train_id_to_string_map)}")

# =========================================================
# 2. 数据加载与预处理 (只运行一次)
# =========================================================
print("\n[Step 2/5] 加载并预处理数据...")
NB_CLASSES = 1173
LENGTH = 5000
INPUT_SHAPE = (LENGTH, 1)

X_train_raw, y_train_raw, X_valid_raw, y_valid_raw, X_test_raw, y_test_raw = LoadDataNoDefCW()

K.set_image_data_format("channels_last")


# 预处理函数化，防止变量污染
def preprocess_data(X, y, to_categorical=True):
    X = safe_log(X.astype('float32'))
    X = X[:, :, np.newaxis]
    y = y.astype('float32')
    if to_categorical:
        y = tf.keras.utils.to_categorical(y, NB_CLASSES)
    return X, y


X_train, y_train = preprocess_data(X_train_raw, y_train_raw)
X_valid, y_valid = preprocess_data(X_valid_raw, y_valid_raw)
# 注意：测试集 y 保持 categorical 用于 evaluate，但也需要保留 index 用于 f1_score
X_test, y_test = preprocess_data(X_test_raw, y_test_raw)
y_test_ids = np.argmax(y_test, axis=1)  # 预先计算好真实标签ID

print(f"   训练集: {X_train.shape[0]}, 验证集: {X_valid.shape[0]}, 测试集: {X_test.shape[0]}")

# =========================================================
# 3. 循环训练 (多次运行取平均)
# =========================================================
NB_RUNS = 5  # 设定重复运行次数
NB_EPOCH = 60  # 设定 Epoch
BATCH_SIZE = 160
VERBOSE = 2

# 累加器：Key=字符串标签, Value=列表[score1, score2, ...]
f1_accumulator = {label: [] for label in train_id_to_string_map.values()}

print(f"\n[Step 3/5] 开始 {NB_RUNS} 次重复实验...")

for run in range(NB_RUNS):
    print(f"\n---> 开始第 {run + 1} / {NB_RUNS} 次运行 <---")

    # 3.1 清理显存与重置图 (至关重要，否则显存会炸)
    K.clear_session()

    # 3.2 设置随机种子 (让每次初始化稍有不同，或者固定种子看你的需求)
    # 这里我们改变种子以探索初始化的鲁棒性
    current_seed = run * 10
    random.seed(current_seed)
    tf.random.set_seed(current_seed)
    np.random.seed(current_seed)

    # 3.3 构建与编译模型
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = DFNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
        OPTIMIZER = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
                      metrics=["accuracy"])  # TopK 可以去掉以节省一点点计算资源

    # 3.4 训练
    history = model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE, epochs=NB_EPOCH,
                        verbose=VERBOSE, validation_data=(X_valid, y_valid))

    # 3.5 预测与计算 F1
    print(f"   正在计算第 {run + 1} 次运行的 F1 分数...")
    y_pred_probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred_ids = np.argmax(y_pred_probs, axis=1)

    # 计算当前轮次的 F1 (数组)
    current_run_f1s = f1_score(y_test_ids, y_pred_ids, average=None)

    # 将分数存入累加器
    for train_id, score in enumerate(current_run_f1s):
        if train_id in train_id_to_string_map:
            label_str = train_id_to_string_map[train_id]
            f1_accumulator[label_str].append(score)

# =========================================================
# 4. 计算平均值并保存
# =========================================================
print("\n[Step 4/5] 计算所有轮次的平均 F1 分数...")

final_avg_f1_mapping = {}

for label_str, scores_list in f1_accumulator.items():
    if len(scores_list) > 0:
        # 计算平均值
        avg_score = np.mean(scores_list)
        # 强制转为 float (numpy类型有时候 save 会有兼容性问题，虽然 npz 没问题)
        final_avg_f1_mapping[str(label_str)] = float(avg_score)
    else:
        # 理论上不会发生，除非该类在测试集中没出现
        final_avg_f1_mapping[str(label_str)] = 0.0

# =========================================================
# 5. 保存结果
# =========================================================
save_path = '../preprocessed_data/df_class_f1_scores_avg.npz'
print(f"\n[Step 5/5] 保存最终平均结果至: {save_path}")

try:
    np.savez(save_path, **final_avg_f1_mapping)
    print("✅ 保存成功！")
    print(f"   共保存 {len(final_avg_f1_mapping)} 个类别。")
    print(f"   实验重复次数: {NB_RUNS}")

    # 打印几个示例看看
    print("\nTop 3 示例结果:")
    sorted_sample = sorted(final_avg_f1_mapping.items(), key=lambda x: x[1], reverse=True)[:3]
    for k, v in sorted_sample:
        print(f"   {k}: {v:.4f}")

except Exception as e:
    print(f"❌ 保存失败: {e}")

print("\n所有任务完成。")
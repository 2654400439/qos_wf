import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. 数据预处理 (只做一次，保证映射一致)
# ==========================================

# 读取 CSV 文件
file_path = '../preprocessed_data/finewp_features_k_3.csv'  # 替换为你的 CSV 文件路径
data = pd.read_csv(file_path, encoding='gbk', header=None)

original_len = len(data)
# 去掉含NaN的行
data_clean = data.dropna()
clean_len = len(data_clean)

print(f"原始样本数: {original_len}")
print(f"清理后样本数: {clean_len}")
print(f"删除了 {original_len - clean_len} 行包含NaN的样本")

# 提取特征和标签
X = data_clean.iloc[:, :-1].values
y = data_clean.iloc[:, -1].values

# 统计每个类别的样本数量
label_counts = Counter(y)

# 筛选出样本数不少于10的类别
valid_labels = [label for label, count in label_counts.items() if count >= 10]
print(f"满足样本数筛选的类别数: {len(valid_labels)}")

# 如果大于1600类，随机选择1600类 (固定种子选择类别，保证任务一致)
np.random.seed(42)
if len(valid_labels) > 1600:
    selected_labels = np.random.choice(valid_labels, size=1600, replace=False)
else:
    selected_labels = valid_labels

selected_labels_set = set(selected_labels)

# 保留属于 selected_labels 的样本
mask = [label in selected_labels_set for label in y]
X_filtered = X[mask]
y_filtered = y[mask]

# 将 y 转换为整数标签
label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(y_filtered)

num_classes = len(label_encoder.classes_)
print(f"最终保留样本数: {len(X_filtered)}")
print(f"最终类别数: {num_classes}")

# ==========================================
# 2. 多次运行循环 (5次)
# ==========================================

n_runs = 5
# 初始化一个数组用于累加每次的F1分数，长度等于类别数
total_f1_scores = np.zeros(num_classes)
# 初始化变量记录平均准确率
total_accuracy = 0
total_top5_accuracy = 0

print(f"\n开始执行 {n_runs} 次训练与评估...\n")

for i in range(n_runs):
    print(f"--- 第 {i + 1} 次运行 ---")

    # 划分训练集和测试集
    # 关键点：random_state=42+i，确保每次划分的数据集都不一样，但又是可复现的
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_int,
        train_size=10 * 1173,
        stratify=y_int,
        random_state=42 + i
    )

    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 初始化随机森林 (也可以让随机森林内部种子也变化，或者固定以观察数据划分的影响)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # 训练
    clf.fit(X_train, y_train)

    # 预测
    y_pred = clf.predict(X_test)

    # --- 计算当前轮次指标 ---

    # 1. 准确率
    acc = accuracy_score(y_test, y_pred)
    total_accuracy += acc
    print(f"  Accuracy: {acc:.4f}")

    # 2. Top-5 准确率
    probs = clf.predict_proba(X_test)
    top5_preds = np.argsort(probs, axis=1)[:, -5:]
    correct_top5 = sum(y_true in top5 for y_true, top5 in zip(y_test, top5_preds))
    top5_acc = correct_top5 / len(y_test)
    total_top5_accuracy += top5_acc
    print(f"  Top-5 Accuracy: {top5_acc:.4f}")

    # 3. 计算本轮各类别 F1 分数
    # labels=np.arange(num_classes) 确保即使测试集中某些类没出现，也会返回0分，保持数组长度对齐
    f1_current = f1_score(y_test, y_pred, labels=np.arange(num_classes), average=None)

    # 累加到总分
    total_f1_scores += f1_current

# ==========================================
# 3. 计算平均值并保存结果
# ==========================================

print("\n" + "=" * 30)
print("所有轮次结束，正在计算平均结果...")

# 计算平均指标
avg_accuracy = total_accuracy / n_runs
avg_top5_accuracy = total_top5_accuracy / n_runs
avg_class_f1_scores = total_f1_scores / n_runs

print(f"平均 Accuracy: {avg_accuracy:.4f}")
print(f"平均 Top-5 Accuracy: {avg_top5_accuracy:.4f}")

# 构建 {字符串标签: 平均F1分数} 的映射
# label_encoder.classes_ 里的顺序和 f1_score 输出的顺序是一致的
class_names = label_encoder.classes_
class_f1_mapping = dict(zip(class_names, avg_class_f1_scores))

# 保存为 .npz 文件
save_path = '../preprocessed_data/finewp_class_f1_scores_avg.npz'  # 文件名我也稍微改了一下加了_avg
np.savez(save_path, **class_f1_mapping)

print(f"已成功将 {n_runs} 次运行的平均各类别 F1 分数保存至: {save_path}")

# 打印几个示例看看
print("前5个类别的平均F1分数示例:")
for k, v in list(class_f1_mapping.items())[:5]:
    print(f"  {k}: {v:.4f}")
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from collections import Counter


# 1. 加载原始数据
with open('../preprocessed_data/DF_X_with_size.pkl', 'rb') as f:
    X_array = pickle.load(f)

with open('../preprocessed_data/DF_y_with_size.pkl', 'rb') as f:
    y_array = pickle.load(f)

# 2. 统计每个类别的样本数
class_counts = Counter(y_array)

# 3. 丢弃样本数 < 10 的类别
eligible_classes = [cls for cls, count in class_counts.items() if count >= 10]

print(f"满足条件的类别数：{len(eligible_classes)}")


# 6. 将原始类别 ID 映射为 0~1599 的新类别 ID
class_list_sorted = sorted(eligible_classes)  # 确保映射是确定性的
class_to_newid = {old: new for new, old in enumerate(class_list_sorted)}
y_filtered_mapped = np.array([class_to_newid[y] for y in y_array])

# 7. 输出信息
print("最终样本数：", len(y_filtered_mapped))
print("最终类别数：", len(set(y_filtered_mapped)))



X_train, X_temp, y_train, y_temp = train_test_split(X_array, y_filtered_mapped, train_size=10*1173, random_state=0, stratify=y_filtered_mapped)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, train_size=2*1173, random_state=0, stratify=y_temp)


print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_valid.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

with open('../preprocessed_data/DF_X_train_with_size_10.pkl', 'wb') as f:
    pickle.dump(X_train, f)
with open('../preprocessed_data/DF_y_train_with_size_10.pkl', 'wb') as f:
    pickle.dump(y_train, f)

with open('../preprocessed_data/DF_X_valid_with_size_10.pkl', 'wb') as f:
    pickle.dump(X_valid, f)
with open('../preprocessed_data/DF_y_valid_with_size_10.pkl', 'wb') as f:
    pickle.dump(y_valid, f)

with open('../preprocessed_data/DF_X_test_with_size_10.pkl', 'wb') as f:
    pickle.dump(X_test, f)
with open('../preprocessed_data/DF_y_test_with_size_10.pkl', 'wb') as f:
    pickle.dump(y_test, f)
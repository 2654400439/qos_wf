import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def normalize_with_direction(data):
    # 保留方向信息
    direction = np.sign(data)

    # 取绝对值忽略方向信息
    abs_data = np.abs(data)

    # 进行全局归一化
    scaler = MinMaxScaler()
    abs_data_normalized = scaler.fit_transform(abs_data)

    # 将方向信息映射回去
    data_normalized = abs_data_normalized * direction

    return data_normalized


# Load data for non-defended dataset for CW setting
def LoadDataNoDefCW():
    print("Loading non-defended dataset for closed-world scenario")

    # Load training data
    with open('../preprocessed_data/DF_X_train_with_size_10_h3.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle))
    with open('../preprocessed_data/DF_y_train_with_size_10_h3.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle))

    # Load validation data
    with open('../preprocessed_data/DF_X_valid_with_size_10_h3.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle))
    with open('../preprocessed_data/DF_y_valid_with_size_10_h3.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle))

    # Load testing data
    with open('../preprocessed_data/DF_X_test_with_size_10_h3.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle))
    with open('../preprocessed_data/DF_y_test_with_size_10_h3.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle))

    print("Data dimensions:")
    print("X: Training data's shape : ", X_train.shape)
    print("y: Training data's shape : ", y_train.shape)
    print("X: Validation data's shape : ", X_valid.shape)
    print("y: Validation data's shape : ", y_valid.shape)
    print("X: Testing data's shape : ", X_test.shape)
    print("y: Testing data's shape : ", y_test.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test




# Load data for non-defended dataset for OW training
def LoadDataNoDefOW_Training():
    print("Loading non-defended dataset for open-world scenario for training")
    # Point to the directory storing data
    dataset_dir = '../dataset/OpenWorld/NoDef/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_NoDef.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_train_NoDef.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle))

    # Load validation data
    with open(dataset_dir + 'X_valid_NoDef.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_valid_NoDef.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle))

    print("Data dimensions:")
    print("X: Training data's shape : ", X_train.shape)
    print("y: Training data's shape : ", y_train.shape)
    print("X: Validation data's shape : ", X_valid.shape)
    print("y: Validation data's shape : ", y_valid.shape)

    return X_train, y_train, X_valid, y_valid

# Load data for non-defended dataset for OW evaluation
def LoadDataNoDefOW_Evaluation():
    print("Loading non-defended dataset for open-world scenario for evaluation")
    # Point to the directory storing data
    dataset_dir = '../dataset/OpenWorld/NoDef/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load testing data
    with open(dataset_dir + 'X_test_Mon_NoDef.pkl', 'rb') as handle:
        X_test_Mon = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_test_Mon_NoDef.pkl', 'rb') as handle:
        y_test_Mon = np.array(pickle.load(handle))
    with open(dataset_dir + 'X_test_Unmon_NoDef.pkl', 'rb') as handle:
        X_test_Unmon = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_test_Unmon_NoDef.pkl', 'rb') as handle:
        y_test_Unmon = np.array(pickle.load(handle))

    return X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon




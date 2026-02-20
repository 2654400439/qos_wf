import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn import metrics
import random
import sys
import pathlib
from sklearn.feature_selection import RFE
from functools import partial
import multiprocessing as mp
import pandas
import sklearn
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import csv

sys.path.append("../..")


K_FOLDS = 5  # 交叉验证的折数
TEST_PERCENTAGE = 0.2  # 测试集的百分比
KF_RFE_NFEATURES_TO_SELECT = 50  # RFE 选择的特征数
KF_RFE_STEPS = 100  # RFE 的步长
N_TREES = 30  # 随机森林的树数
N_CLASSES = -1  # 类别数，默认值为 -1 表示不限制类别数




def get_k_neighbor(params):
    y_train_predicted, y_test_predicted, y_train, K = (
        params[0],
        params[1],
        params[2],
        params[3],
    )

    atile = np.tile(y_test_predicted, (y_train_predicted.shape[0], 1))
    dists = np.sum(atile != y_train_predicted, axis=1)
    k_neighbors = y_train[np.argsort(dists)[:K]]

    return k_neighbors


def parallel_get_k_neighbors(
        y_train_predicted, y_test_predicted, y_train, K=3, n_jobs=20
):
    n = len(y_test_predicted)

    Y_train_predicted = [y_train_predicted] * n
    Y_train = [y_train] * n
    Ks = [K] * n

    zipped = zip(Y_train_predicted, y_test_predicted, Y_train, Ks)

    pool = mp.Pool(n_jobs)
    neighbors = pool.map(get_k_neighbor, zipped)
    return np.array(neighbors)


def random_forest(features_names, X_train, y_train, X_test, y_test, n_trees=1000, rfe_nfeatures=800, rfe_steps=10,
                  n_classes=-1):
    clf = RandomForestClassifier(n_jobs=-1, n_estimators=n_trees, random_state=9)
    selector = RFE(estimator=clf, n_features_to_select=rfe_nfeatures, step=rfe_steps)

    selector = selector.fit(X_train, y_train)
    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)

    clf2 = RandomForestClassifier(n_jobs=-1, n_estimators=n_trees, random_state=9)
    clf2.fit(X_train, y_train)
    y_pred = clf2.predict(X_test)
    predicted_probas = clf2.predict_proba(X_test)

    min_n_samples = 0
    c = {}
    for y in y_test:
        if not y in c:
            c[y] = 0
        c[y] += 1
    min_n_samples = min([x[1] for x in c.items()])
    n_uniq_classes = len(list(set(y_test)))

    scores = dict(
        n_classes=n_uniq_classes,
        min_n_samples=min([x[1] for x in c.items()]),
        max_n_samples=max([x[1] for x in c.items()]),
        accuracy=metrics.accuracy_score(y_test, y_pred),
        precision=metrics.precision_score(y_test, y_pred, average='macro'),
        recall=metrics.recall_score(y_test, y_pred, average='macro'),
        f1score=metrics.f1_score(y_test, y_pred, average='macro'),
    )

    if n_classes > -1 and n_uniq_classes != n_classes:
        y_test_ext = [y for y in y_test]
        y_pred_ext = [y for y in y_pred]

        n_to_add = n_classes - n_uniq_classes
        next_value = 900  # arbitrary, just not a url
        i = 0
        while i < n_to_add:
            y_test_ext.extend([next_value + i] * min_n_samples)
            y_pred_ext.extend([-1] * min_n_samples)
            i += 1

        c = {}
        for y in y_test_ext:
            if not y in c:
                c[y] = 0
            c[y] += 1

        print(f"{n_classes} but only found {n_uniq_classes}")

        scores['n_classes_corr'] = len(list(set(y_test_ext)))
        scores['accuracy_corr'] = metrics.accuracy_score(y_test_ext, y_pred_ext)
        scores['precision_corr'] = metrics.precision_score(y_test_ext, y_pred_ext, average='macro')
        scores['recall_corr'] = metrics.recall_score(y_test_ext, y_pred_ext, average='macro')
        scores['f1score_corr'] = metrics.f1_score(y_test_ext, y_pred_ext, average='macro')

        print(f"Scores were {scores['f1score']} but were corrected to {scores['f1score_corr']}.")

    selected_features = []
    i = 0
    while i < len(features_names):
        if selector.support_[i] == 1:
            selected_features.append(features_names[i])
        i += 1

    feature_importance = sorted(zip(clf2.feature_importances_, selected_features), reverse=True)

    return scores, feature_importance, y_pred, predicted_probas


def rf_folds(X, y, feature_names, rfe_nfeatures=50, rfe_steps=100, n_trees=30, n_classes=-1):
    sss = StratifiedShuffleSplit(n_splits=3, train_size=10*1173, random_state=9)
    # sss = StratifiedShuffleSplit(n_splits=K_FOLDS, test_size=TEST_PERCENTAGE, random_state=0)

    print("Number of classes", len(set(y)))
    print("Number of features", len(X[0]))
    print("Number of samples", len(X))
    print("Number of labels", len(y))

    scores = []
    y_test_all = []
    y_pred_all = []
    feature_ranks = dict()

    i = 0
    for train_index, test_index in sss.split(X, y):
        print("Fold", i, end="\r")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        s, features, y_pred, _ = random_forest(feature_names, X_train, y_train, X_test, y_test, n_trees=n_trees,
                                               rfe_nfeatures=rfe_nfeatures, rfe_steps=rfe_steps, n_classes=n_classes)

        y_test_all.extend(y_test)
        y_pred_all.extend(y_pred.tolist())

        for proba, feature in features:
            if not feature in feature_ranks:
                feature_ranks[feature] = []

            feature_ranks[feature].append(proba)

        scores.append(s)
        i += 1

    # average scores
    score = {k: (np.mean([value[k] for value in scores]), np.std([value[k] for value in scores])) for k in scores[0]}

    # average features importance
    for f in feature_ranks:
        if len(feature_ranks[f]) < K_FOLDS:
            feature_ranks[f].extend([0] * (K_FOLDS - len(feature_ranks[f])))

    features_and_percentages = []
    for f in feature_ranks:
        features_and_percentages.append((f, np.mean(feature_ranks[f]), np.std(feature_ranks[f])))

    return score, features_and_percentages, y_test_all, y_pred_all


def trim(elements, n):
    if len(elements) >= n:
        elements[:n] = True
    return elements


def trim_df(X, y, num_insts):
    """Return a dataframe with the same number of instances per class.
    The dataframe, `df`, has a field with the class id called `class_label`.
    """
    X2 = X.copy()
    y2 = y.copy()

    X['selected'] = False  # initialize all instances to not selected
    classes = y2.groupby('class_label')  # group instances by class
    trim_part = partial(trim, n=num_insts)  # partial trim to n=NUM_INSTS
    X['selected'] = classes.selected.transform(trim_part)  # mark as selected
    selected = X[X2.selected]  # get the selected instances
    return selected


def sample_classes(df, classes=None):
    if type(classes) is int:
        sample = random.sample(df.class_label.unique(), classes)
    elif type(classes) is list:
        sample = classes
    else:
        raise Exception("Type of classes not recognized.")
    selected_classes = df.class_label.isin(sample)
    return df[selected_classes]


def labels_strings_to_ids(labels):
    mapping = dict()
    i = 0
    for l in labels:
        if l in mapping:
            continue
        mapping[l] = i
        i += 1

    labels2 = [mapping[l] for l in labels]

    return labels2, mapping


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np


def evaluate_model_with_class_f1(X, y, feature_names, test_size=0.5, n_trees=30, rfe_nfeatures=50, rfe_steps=100,
                                 random_state=42):
    """
    在函数内部进行数据集切分，训练模型并评估，输出每个类别的F1值。

    参数:
        X: 完整的特征数据 (numpy array)
        y: 完整的标签数据 (numpy array)
        feature_names: 特征名称列表 (list)
        test_size: 测试集占比 (float, default=0.2)
        n_trees: 随机森林中树的数量 (int)
        rfe_nfeatures: RFE选择的特征数量 (int)
        rfe_steps: RFE每轮剔除的特征数量 (int)
        random_state: 随机种子 (int)

    返回:
        scores: 包含整体评估指标的字典
        class_f1_scores: 每个类别的F1分数 (numpy array)
        selected_features: 被选中的特征名称 (list)
    """
    random_state = random.randint(0, 10000)
    # 数据集切分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        stratify=y)

    # 初始化随机森林分类器
    clf = RandomForestClassifier(n_jobs=-1, n_estimators=n_trees, random_state=random_state)

    # 使用RFE进行特征选择
    selector = RFE(estimator=clf, n_features_to_select=rfe_nfeatures, step=rfe_steps)
    selector = selector.fit(X_train, y_train)

    # 转换训练集和测试集特征
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    # 重新训练模型
    clf_final = RandomForestClassifier(n_jobs=-1, n_estimators=n_trees, random_state=random_state)
    clf_final.fit(X_train_selected, y_train)

    # 预测测试集
    y_pred = clf_final.predict(X_test_selected)

    # 计算整体评估指标
    scores = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='macro'),
        "recall": recall_score(y_test, y_pred, average='macro'),
        "f1score": f1_score(y_test, y_pred, average='macro')
    }

    # 计算每个类别的F1分数
    class_f1_scores = f1_score(y_test, y_pred, average=None)

    # 获取被选中的特征名称
    selected_features = [feature_names[i] for i in range(len(feature_names)) if selector.support_[i]]

    return scores, class_f1_scores, selected_features


def run_multiple_evaluations(X, y, feature_names, n_runs=5, **kwargs):
    """
    多次运行模型评估函数，并对每个类别的F1分数取平均值。

    参数:
        X: 完整的特征数据 (numpy array)
        y: 完整的标签数据 (numpy array)
        feature_names: 特征名称列表 (list)
        n_runs: 运行次数 (int, default=5)
        **kwargs: 传递给evaluate_model_with_class_f1的其他参数

    返回:
        avg_scores: 平均的整体评估指标 (dict)
        avg_class_f1_scores: 每个类别的平均F1分数 (numpy array)
        all_selected_features: 每次运行选中的特征 (list of lists)
    """
    # 存储每次运行的结果
    all_scores = []
    all_class_f1_scores = []
    all_selected_features = []

    # 多次运行评估函数
    for i in range(n_runs):
        scores, class_f1_scores, selected_features = evaluate_model_with_class_f1(X, y, feature_names, **kwargs)
        all_scores.append(scores)
        all_class_f1_scores.append(class_f1_scores)
        all_selected_features.append(selected_features)

    # 计算整体评估指标的平均值
    avg_scores = {
        key: np.mean([score[key] for score in all_scores])
        for key in all_scores[0].keys()
    }

    # 计算每个类别F1分数的平均值
    avg_class_f1_scores = np.mean(all_class_f1_scores, axis=0)

    return avg_scores, avg_class_f1_scores, all_selected_features



def run(dataset_npy):
    print("Running attack on {}".format(dataset_npy))

    dic = np.load(dataset_npy, allow_pickle=True).item()

    feature_names = dic["feature_names"]
    X_raw = np.array(dic["features"])
    Y_raw = np.array(dic["labels"])
    # 过滤待处理的文件
    with open('../../../domain_fine_qoe_pkt.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        domain_list = [row[0] for row in reader]
    # 执行过滤
    mask = np.isin(Y_raw, domain_list)

    # 使用布尔掩码过滤数据
    X = X_raw[mask]
    Y = Y_raw[mask]

    print(f"原始样本数: {len(X_raw)}")
    print(f"过滤后样本数: {len(X)}")
    # print(f"保留的标签: {set(Y)}")

    y_str = np.array([label for label in Y])

    y_i, mapping = labels_strings_to_ids(y_str)

    y_i = np.array(y_i)

    if len(X) == 0:
        print(f"Warning: empty dataset {dataset_npy}")
        score = dict(
            n_classes=0,
            min_n_samples=0,
            max_n_samples=0,
            accuracy=[0, 0],
            precision=[0, 0],
            recall=[0, 0],
            f1score=[0, 0]
        )
        return dict(score=score, features=[])

    print(f"Loaded {len(X)} rows, {len(feature_names)} feature names for {len(X[0])} features")
    # score, features_and_percentages, y_test_all, y_pred_all = rf_folds(X, y_i, feature_names,
    #                                                                    rfe_nfeatures=KF_RFE_NFEATURES_TO_SELECT,
    #                                                                    rfe_steps=KF_RFE_STEPS, n_trees=N_TREES,
    #                                                                    n_classes=N_CLASSES)

    # 调用多次评估函数
    avg_scores, avg_class_f1_scores, all_selected_features = run_multiple_evaluations(
        X, y_i, feature_names, n_runs=5, test_size=0.5, n_trees=30, rfe_nfeatures=50, rfe_steps=100
    )

    # 输出结果
    print("平均整体评估指标:", avg_scores)
    # # print("每个类别的平均F1分数:", avg_class_f1_scores)
    # for label_id, f1_score in enumerate(avg_class_f1_scores):
    #     label = [key for key, value in mapping.items() if value == label_id][0]
    #     print(f"类别 '{label}' 的平均 F1 分数: {f1_score:.4f}")

    # 构建类别标签与 F1 分数的映射字典
    class_f1_mapping = {label: f1_score for label_id, f1_score in enumerate(avg_class_f1_scores)
                        for label, value in mapping.items() if value == label_id}

    # 保存为 .npz 文件
    np.savez('../preprocessed_data/kfp_class_f1_scores.npz', **class_f1_mapping)




if __name__ == "__main__":
    run('../myself/kfp_features_all.npy')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


# ==========================================
# 第一部分：配置与辅助函数
# ==========================================
# 设置绘图风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'


def load_npz_scores(file_path):
    """辅助函数：安全加载单个NPZ文件并返回 {domain: score} 字典"""
    try:
        with np.load(file_path, allow_pickle=True) as npz_data:
            keys = list(npz_data.keys())
            data_map = None

            # 尝试提取数据
            if 'arr_0' in keys:
                content = npz_data['arr_0']
                if content.ndim == 0:
                    data_map = content.item()

            if data_map is None:
                data_map = {}
                for k in keys:
                    val = npz_data[k]
                    data_map[k] = val.item() if val.ndim == 0 else val

            # 确保是字典且值是标量
            clean_map = {}
            if isinstance(data_map, dict):
                for k, v in data_map.items():
                    # 如果是数组取均值，如果是标量直接用
                    score = np.mean(v) if isinstance(v, (np.ndarray, list)) else v
                    clean_map[k] = score
            return clean_map

    except Exception as e:
        print(f"加载文件 {file_path} 失败: {e}")
        return {}


# ==========================================
# 第二部分：数据加载与预处理
# ==========================================

def load_and_process_data():
    # 1. 定义三个模型对应的文件路径
    model_files = {
        "k-FP": "../preprocessed_data/kfp_class_f1_scores.npz",
        "DF": "../preprocessed_data/df_class_f1_scores_avg.npz",
        "STAR": "../preprocessed_data/star_class_f1_scores.npz",
        "FineWP": "../preprocessed_data/finewp_class_f1_scores_avg.npz",
        "RF": "../preprocessed_data/rf_class_f1_scores.npz"
    }

    # 2. 分别加载每个模型的数据
    dfs = []
    for model_name, file_path in model_files.items():
        if os.path.exists(file_path):
            scores_map = load_npz_scores(file_path)
            # 转换为 DataFrame: Domain, F1_ModelName
            df_model = pd.DataFrame(list(scores_map.items()), columns=['Domain', f'F1_{model_name}'])
            dfs.append(df_model)
            print(f"[{model_name}] 加载成功，共 {len(df_model)} 个网站")
        else:
            print(f"警告: 文件不存在 {file_path}")

    if not dfs:
        print("未加载到任何分类结果数据")
        return None, None

    # 3. 合并所有模型的分数表
    # 使用 outer join 保证尽量保留数据，或者 inner join 保证只有共有的才分析
    df_f1 = dfs[0]
    for i in range(1, len(dfs)):
        df_f1 = pd.merge(df_f1, dfs[i], on='Domain', how='inner')

    print(f"模型数据合并完成，共 {len(df_f1)} 个共有网站")

    # 4. 加载 QoS CSV 文件
    csv_path = "../preprocessed_data/domain_metrics_full.csv"
    if not os.path.exists(csv_path):
        print(f"CSV文件不存在: {csv_path}")
        return None, None

    df_qos = pd.read_csv(csv_path)

    # 5. QoS 数据清洗
    numeric_cols = df_qos.select_dtypes(include=[np.number]).columns.tolist()
    # 排除掉无关的列（如果有 Label 或 Rank 之类的）
    cols_to_exclude = ['Rank', 'Label']
    numeric_cols = [c for c in numeric_cols if c not in cols_to_exclude]

    for col in numeric_cols:
        if df_qos[col].isnull().any():
            median_val = df_qos[col].median()
            df_qos[col] = df_qos[col].fillna(median_val)

    # 6. 最终合并
    df_merged = pd.merge(df_f1, df_qos, on="Domain", how="inner")
    print(f"最终合并有效数据量: {len(df_merged)} 行")

    return df_merged, numeric_cols


# ==========================================
# 第三部分：计算相关性与可视化 (偏好热力图)
# ==========================================

def analyze_attacker_preference(df, metric_cols):
    if df is None or len(df) == 0:
        return

    # 目标列：即我们在 load 阶段生成的 F1_k-FP, F1_DF, F1_Star
    target_cols = [c for c in df.columns if c.startswith('F1_')]

    if not target_cols:
        print("未找到 F1 分数数据列")
        return

    print(f"正在分析以下攻击模型: {target_cols}")

    # 1. 计算斯皮尔曼相关系数矩阵
    correlation_results = []

    for metric in metric_cols:
        # 跳过方差为0的常量列
        if df[metric].std() == 0:
            continue

        row_res = {'Metric': metric}
        for target in target_cols:
            # 计算相关性
            corr, p_val = spearmanr(df[metric], df[target])
            # 存储格式：去掉前缀 'F1_' 让图表更好看
            model_name = target.replace('F1_', '')
            row_res[model_name] = corr

        correlation_results.append(row_res)

    df_corr = pd.DataFrame(correlation_results)

    # 2. 排序与整理
    # 设置索引以便绘图
    df_corr = df_corr.set_index('Metric')

    # 可选：按照其中一个模型（比如 k-FP）的相关性绝对值排序，或者按平均相关性排序
    # 这里我们按 '平均绝对相关性' 排序，把大家都敏感的放上面
    df_corr['abs_mean'] = df_corr.abs().mean(axis=1)
    df_corr = df_corr.sort_values('abs_mean', ascending=False).drop(columns=['abs_mean'])

    # 3. 绘制“攻击者偏好”热力图 (Attacker Preference Heatmap)
    # 动态调整图片高度：每行 0.4 英寸
    fig_height = max(8, len(df_corr) * 0.4)
    plt.figure(figsize=(7, fig_height))

    # 使用红蓝配色 (RdBu_r): 红=正相关, 蓝=负相关, 白=无相关
    # 调整 vmin/vmax 让颜色对比更明显，一般 0.15 到 0.2 就很有区分度了
    sns.heatmap(df_corr, annot=True, fmt=".3f", cmap="RdBu_r",
                center=0, vmin=-0.2, vmax=0.2,
                linewidths=1, linecolor='white',
                cbar_kws={"label": "Spearman Correlation", "shrink": 0.5})

    plt.title("Attacker Sensitivity: Feature Preference Analysis", fontsize=15, pad=20)
    plt.ylabel("QoS / Runtime / QoE Metrics", fontsize=12)
    plt.xlabel("Fingerprinting Method", fontsize=12)

    # 将 X 轴标签放到底部并水平显示
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.show()

    # 4. 打印每个模型的 Top 3 敏感指标 (正/负)
    print("\n====== Attacker Preference Summary ======")
    for model in df_corr.columns:
        print(f"\nModel: {model}")
        series = df_corr[model].sort_values(ascending=False)
        print("  Top 3 Positive Correlated (Complexity/Volume?):")
        print(series.head(3).to_string())
        print("  Top 3 Negative Correlated (Optimization/Protocol?):")
        print(series.tail(3).to_string())


def analyze_multivariate_importance(df, feature_cols, target_col='F1_Max'):
    """
    使用 XGBoost + SHAP 分析多变量对易识别性的联合影响
    """
    print(f"--- 正在进行多变量回归分析 (Target: {target_col}) ---")

    # 1. 数据准备
    X = df[feature_cols]
    y = df[target_col]

    # 切分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. 训练回归模型 (XGBoost)
    # 我们限制深度，防止过拟合，主要看特征贡献
    model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    # 3. 模型评估 (看看这组指标到底能不能预测 F1)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"\n>>> 模型解释力评估 <<<")
    print(f"R² Score (决定系数): {r2:.4f}")
    print(f"解释: QoS 指标组合能够解释 {r2 * 100:.2f}% 的易识别性方差。")
    print(f"如果 R² < 0.2，说明 QoS 不是主要因素；如果 > 0.5，说明关系很强。")

    # 4. SHAP 值计算 (归因分析)
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    # 5. 可视化

    # 图 1: SHAP Summary Plot (Beeswarm)
    # 这张图是目前顶会最喜欢的：展示了特征的重要性 + 正负影响方向
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"SHAP Value Impact on {target_col} (Systematic Analysis)", fontsize=14)
    plt.tight_layout()
    plt.show()

    # 图 2: Bar Plot (纯重要性排序)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("Feature Importance Ranking (multivariate)", fontsize=14)
    plt.tight_layout()
    plt.show()

    return model, r2





# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    df_final, metric_columns = load_and_process_data()
    # analyze_attacker_preference(df_final, metric_columns)

    # 使用示例 (假设 df_final 已经有了)
    feature_cols = [c for c in metric_columns if c in df_final.columns]
    analyze_multivariate_importance(df_final, feature_cols, target_col='F1_Star') # 针对 k-FP
    # analyze_multivariate_importance(df_final, feature_cols, target_col='F1_DF')   # 针对 DF
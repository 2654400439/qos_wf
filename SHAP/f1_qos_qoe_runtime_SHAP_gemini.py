import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ==========================================
# 配置部分
# ==========================================
# 绘图风格设置 (符合论文标准)
plt.rcParams['font.family'] = 'sans-serif'  # 论文建议先用 sans-serif, 投稿时再换 Times
plt.rcParams['axes.unicode_minus'] = False
sns.set_context("paper", font_scale=1.2)


def load_and_process_data():
    print("开始加载数据...")

    # 1. 加载 5 个 NPZ 文件 (包含了所有攻击模型的结果)
    npz_files = [
        "../preprocessed_data/kfp_class_f1_scores.npz",
        "../preprocessed_data/df_class_f1_scores_avg.npz",
        "../preprocessed_data/star_class_f1_scores.npz",
        "../preprocessed_data/finewp_class_f1_scores_avg.npz",
        "../preprocessed_data/rf_class_f1_scores.npz"
    ]

    # 用于存储合并后的分数: {domain: [score1, score2, ...]}
    combined_scores = {}

    for file_path in npz_files:
        try:
            with np.load(file_path, allow_pickle=True) as npz_data:
                keys = list(npz_data.keys())
                data_map = None

                # 策略: 检查是否有一个主键（如 'arr_0'）存储了整个字典
                if 'arr_0' in keys:
                    content = npz_data['arr_0']
                    if content.ndim == 0:
                        data_map = content.item()

                # 如果没有命中 'arr_0'，尝试直接读取所有键
                if data_map is None:
                    data_map = {}
                    for k in keys:
                        val = npz_data[k]
                        # 提取 0-d array 的标量值
                        if val.ndim == 0:
                            data_map[k] = val.item()
                        else:
                            data_map[k] = val

            # 将提取出的数据存入 combined_scores
            if data_map:
                for domain, score in data_map.items():
                    if domain not in combined_scores:
                        combined_scores[domain] = []

                    # 确保分数是标量
                    if isinstance(score, np.ndarray) and score.size == 1:
                        score = score.item()
                    combined_scores[domain].append(score)

        except Exception as e:
            print(f"加载文件 {file_path} 失败: {e}")
            continue

    # [核心修复] 计算 F1_Mean 和 F1_Max
    f1_data = []
    for domain, scores in combined_scores.items():
        if len(scores) > 0:
            f1_data.append({
                "Domain": domain,
                "F1_Mean": np.mean(scores),  # 计算均值
                "F1_Max": np.max(scores)  # 计算最大值
            })

    df_f1 = pd.DataFrame(f1_data)
    print(f"成功加载易识别性数据，共 {len(df_f1)} 个网站。")

    # 2. 加载 QoS CSV
    csv_path = "../preprocessed_data/domain_metrics_full.csv"
    try:
        df_qos = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误: 找不到 QoS 文件 {csv_path}")
        return pd.DataFrame(), []

    # 3. 数据合并 (Inner Join)
    df_merged = pd.merge(df_f1, df_qos, on="Domain", how="inner")

    # 4. 提取纯特征列
    #    这里我们自动筛选数值列
    feature_cols = df_merged.select_dtypes(include=[np.number]).columns.tolist()

    # [重要] 从特征列表中移除 Target 列 (F1_Mean, F1_Max)
    # 否则做偏相关或 VIF 时会把自己和自己做回归
    for target in ['F1_Mean', 'F1_Max']:
        if target in feature_cols:
            feature_cols.remove(target)

    # 清洗：填充缺失值
    for col in feature_cols:
        if df_merged[col].isnull().any():
            df_merged[col] = df_merged[col].fillna(df_merged[col].median())

    print(f"合并后用于分析的数据量: {len(df_merged)} 行。")
    return df_merged, feature_cols


# ==========================================
# 2. 核心分析类 (封装 SHAP 逻辑)
# ==========================================
class FingerprintExplainer:
    def __init__(self, df, feature_cols, target_col='F1_Mean'):
        self.df = df
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.model = None
        self.explainer = None
        self.shap_values = None
        self.X = df[feature_cols]
        self.y = df[target_col]

    def train(self):
        print(f"--- Training XGBoost Regressor on {len(self.df)} samples ---")
        # 训练集测试集划分
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # XGBoost 参数调优 (针对小数据集防止过拟合)
        self.model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,  # 限制深度，强迫模型选主要特征
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        )
        self.model.fit(self.X_train, self.y_train)

        # 评估
        r2 = self.model.score(self.X_test, self.y_test)
        print(f"Model R² Score: {r2:.4f} (Interpretation Power)")

        # 计算 SHAP
        print("Calculating SHAP values...")
        self.explainer = shap.Explainer(self.model, self.X_train)
        # 计算全量数据的 SHAP (为了后续分层分析方便)
        self.shap_values = self.explainer(self.X)

    def plot_global_summary(self):
        """实验一：全局 SHAP Summary (Beeswarm)"""
        plt.figure(figsize=(10, 6))
        plt.title("Global Feature Attribution (Mechanism Ranking)", fontsize=14)
        # max_display 控制显示的特征数量，建议 10-15 个
        shap.summary_plot(self.shap_values, self.X, max_display=12, show=False)
        plt.tight_layout()
        plt.savefig("SHAP_summary.pdf", bbox_inches='tight')
        plt.show()

    def plot_dependence(self, feature_name):
        """实验二：单特征依赖图 (Non-linear Mechanism)"""
        if feature_name not in self.feature_cols:
            print(f"Feature {feature_name} not found!")
            return

        plt.figure(figsize=(8, 5))
        # interaction_index=None 表示不自动寻找交互特征上色，保持图表简单
        shap.dependence_plot(feature_name, self.shap_values.values, self.X,
                             interaction_index=None, show=False)
        plt.title(f"Mechanism: How {feature_name} affects Leakage", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def analyze_head_vs_tail(self, head_rank=500, tail_rank=10000):
        """实验三：分层归因 (The Optimization Paradox)"""
        if 'Rank' not in self.df.columns:
            print("Error: 'Rank' column missing for stratified analysis.")
            return

        print(f"\n--- Stratified Analysis: Head (Top {head_rank}) vs. Tail (Rank > {tail_rank}) ---")

        # 1. 获取索引
        head_mask = self.df['Rank'] <= head_rank
        tail_mask = self.df['Rank'] > tail_rank

        # 2. 提取对应的 SHAP 值绝对值均值 (Mean Absolute SHAP)
        # shap_values.values 是 numpy array [samples, features]
        shap_head = np.abs(self.shap_values.values[head_mask]).mean(axis=0)
        shap_tail = np.abs(self.shap_values.values[tail_mask]).mean(axis=0)

        # 3. 组装成 DataFrame
        df_comp = pd.DataFrame({
            'Feature': self.feature_cols,
            'Head (Top 500)': shap_head,
            'Tail (Long-tail)': shap_tail
        })

        # 计算相对差异，用于排序 (关注差异最大的特征)
        df_comp['Diff'] = df_comp['Head (Top 500)'] - df_comp['Tail (Long-tail)']
        df_comp = df_comp.sort_values('Head (Top 500)', ascending=False)  # 按 Head 重要性排序

        # 4. 绘图 (双柱对比图)
        df_plot = df_comp.head(10).melt(id_vars='Feature',
                                        value_vars=['Head (Top 500)', 'Tail (Long-tail)'],
                                        var_name='Group', value_name='SHAP Importance')

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_plot, x='SHAP Importance', y='Feature', hue='Group', palette="muted")
        plt.title("The Optimization Paradox: Leakage Drivers Shift", fontsize=14)
        plt.xlabel("mean(|SHAP value|) - Impact on Fingerprintability")
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        return df_comp

    def plot_interaction_analysis(explainer, shap_values, X, feature_x, feature_color):
        """
        绘制交互效应图：观察 feature_color 如何改变 feature_x 的影响
        """
        plt.figure(figsize=(8, 6))

        # interaction_index 参数是关键！它指定了用哪个特征来上色
        shap.dependence_plot(
            feature_x,
            shap_values.values,
            X,
            interaction_index=feature_color,  # 指定交互变量
            show=False,
            alpha=0.8,
            dot_size=20
        )

        plt.title(f"Interaction: Does {feature_color} mitigate {feature_x}?", fontsize=12)
        plt.tight_layout()
        plt.show()


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 准备特征列表 (必须是 VIF 筛选后的!)
    # 这里填入你在 VIF 步骤决定保留的特征
    # [注意] 不要包含 'Rank', 'Label', 'Domain'
    refined_feature_list = [
        'TLS Setup', 'HTTP3 Ratio', 'Total Transferred Bytes',  # Network
        'JS Exec Time', 'Paint Count', 'Layout Count',  # Runtime
        'TTI', 'LCP',  # QoE
        # 'num_img', 'num_uni_domain'  # Static (if kept)
    ]

    df_final, numeric_cols = load_and_process_data()

    # 2. 加载数据 (假设 df_final 是你现在的 dataframe)
    # [重要] 确保 df_final 里有 'Rank' 列！
    # 如果没有 Rank，需要去 merge 原始的 meta data

    explainer = FingerprintExplainer(df_final, refined_feature_list, target_col='F1_Mean')
    explainer.train()

    # 3. 运行三个实验
    # Experiment 1: Global Summary
    explainer.plot_global_summary()

    # Experiment 2: Mechanisms
    # explainer.plot_dependence('JS Exec Time')
    # explainer.plot_dependence('HTTP3 Ratio')

    # Experiment 3: Head vs Tail
    # explainer.analyze_head_vs_tail()

    # explainer.plot_interaction_analysis(explainer, shap_values, X, "JS Exec Time", "HTTP3 Ratio")
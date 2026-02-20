import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def load_and_process_data():
    print("\n--- 开始加载数据 ---")

    # ==========================================
    # 1. 加载 F1 分数 (Target)
    # ==========================================
    file_map = {
        "KFP": "../preprocessed_data/kfp_class_f1_scores.npz",
        "DF": "../preprocessed_data/df_class_f1_scores_avg.npz",
        "Star": "../preprocessed_data/star_class_f1_scores.npz",
        "FineWP": "../preprocessed_data/finewp_class_f1_scores_avg.npz",
        "RF": "../preprocessed_data/rf_class_f1_scores.npz"
    }

    all_attacks_data = {}
    for attack_name, file_path in file_map.items():
        try:
            with np.load(file_path, allow_pickle=True) as npz_data:
                keys = list(npz_data.keys())
                data_map = None
                if 'arr_0' in keys:
                    content = npz_data['arr_0']
                    if content.ndim == 0: data_map = content.item()
                if data_map is None:
                    data_map = {}
                    for k in keys:
                        val = npz_data[k]
                        if val.ndim == 0:
                            data_map[k] = val.item()
                        else:
                            data_map[k] = val
                clean_map = {}
                if data_map:
                    for d, s in data_map.items():
                        if isinstance(s, np.ndarray) and s.size == 1: s = s.item()
                        clean_map[d] = s
                all_attacks_data[attack_name] = clean_map
        except Exception:
            pass

    df_f1 = pd.DataFrame(all_attacks_data)
    df_f1.index.name = 'Domain'
    df_f1.reset_index(inplace=True)

    # 计算均值
    attack_cols = [col for col in file_map.keys() if col in df_f1.columns]
    df_f1['F1_Mean'] = df_f1[attack_cols].mean(axis=1)
    df_f1.dropna(subset=['F1_Mean'], inplace=True)
    print(f"1. 攻击结果加载完成: {len(df_f1)} 个域名")

    # ==========================================
    # 2. 加载 QoS 指标
    # ==========================================
    df_qos = pd.read_csv("../preprocessed_data/domain_metrics_full.csv")
    # 确保格式一致 (去空格，转字符串)
    df_qos['Domain'] = df_qos['Domain'].astype(str).str.strip()
    print(f"2. QoS 数据加载完成: {len(df_qos)} 行")

    # ==========================================
    # 3. [新增] 加载物理统计特征 (IAT, Burst, Rank)
    # ==========================================
    # 请确认这里的文件路径是否正确
    traffic_csv_path = "../preprocessed_data/traffic_physical_features.csv"
    try:
        df_physical = pd.read_csv(traffic_csv_path)
        # 同样做格式清洗
        df_physical['Domain'] = df_physical['Domain'].astype(str).str.strip()
        print(f"3. 物理特征加载完成: {len(df_physical)} 行 (包含 Rank, IAT, Burst 等)")
    except FileNotFoundError:
        print(f"[错误] 找不到文件: {traffic_csv_path}")
        return None, None

    # ==========================================
    # 4. 数据合并 (F1 + QoS + Physical)
    # ==========================================
    # 第一步: 合并 F1 和 QoS
    df_step1 = pd.merge(df_f1, df_qos, on="Domain", how="inner")
    print(f"   -> 合并 F1 + QoS 后: {len(df_step1)} 行")

    # [细节处理] 防止 Rank 列冲突
    # 如果 df_step1 里已经有 Rank，而 df_physical 里也有 Rank，
    # merge 会产生 Rank_x 和 Rank_y。
    # 我们优先使用 df_physical 里的 Rank (因为是你刚生成的)，所以如果 df_step1 有 Rank 先删掉
    if 'Rank' in df_step1.columns and 'Rank' in df_physical.columns:
        df_step1 = df_step1.drop(columns=['Rank'])

    # 第二步: 合并 物理特征
    # 注意：这里不需要转换下划线，直接匹配
    df_final = pd.merge(df_step1, df_physical, on="Domain", how="inner")
    print(f"   -> 合并 Physical Features 后最终: {len(df_final)} 行")

    # ==========================================
    # 5. 清洗与缺失值填充
    # ==========================================
    feature_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()

    # 排除不需要作为输入的列 (Target 和 辅助列)
    # 注意：Rank 既可以是特征也可以是混杂变量，这里暂时保留在 DataFrame 里
    # 但如果是跑回归模型预测 F1，通常要把 Rank 排除，这里先不从 df 删除，只从 feature_cols 列表剔除
    targets_to_exclude = attack_cols + ['F1_Mean', 'F1_Max']
    for target in targets_to_exclude:
        if target in feature_cols:
            feature_cols.remove(target)

    if 'TCP Handshake' in feature_cols:
        feature_cols.remove('TCP Handshake')

    # 填充缺失值
    for col in feature_cols:
        if df_final[col].isnull().any():
            df_final[col] = df_final[col].fillna(df_final[col].median())

    return df_final, attack_cols


def perform_psm_causal_inference(df, treatment_col, outcome_col, confounder_cols):
    """
    执行倾向性得分匹配 (PSM) 以计算平均处理效应 (ATT)
    """
    print(f"\n[Causal Analysis] Treatment: {treatment_col} -> Outcome: {outcome_col}")

    # 1. 数据清洗：提取相关列并去空值
    cols = [treatment_col, outcome_col] + confounder_cols
    data = df[cols].dropna().copy()

    # 2. 定义处理组 (Treatment = 1) 和 对照组 (Treatment = 0)
    # 我们定义 Top 20% 的指标值为 "High" (Treatment Group)
    threshold = data[treatment_col].quantile(0.80)
    data['T'] = (data[treatment_col] > threshold).astype(int)

    print(f"  - Treatment Threshold (>80%): {threshold:.2f}")
    print(f"  - Treated Samples (High Group): {data['T'].sum()}")
    print(f"  - Control Samples (Low Group): {(data['T'] == 0).sum()}")

    # 3. 计算倾向性得分 (Propensity Score Estimation)
    # 我们用混杂变量来预测 "样本属于 High Group 的概率"
    X = data[confounder_cols]
    T = data['T']
    Y = data[outcome_col]

    # 标准化混杂变量 (这对距离计算很重要)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 逻辑回归计算 PS Score
    ps_model = LogisticRegression(solver='liblinear', random_state=42)
    ps_model.fit(X_scaled, T)
    ps_score = ps_model.predict_proba(X_scaled)[:, 1]
    data['ps_score'] = ps_score

    # 4. 进行匹配 (Matching)
    # 策略：为每个 Treatment 样本找 1 个 PS Score 最接近的 Control 样本
    treated_idx = data[data['T'] == 1].index
    control_idx = data[data['T'] == 0].index

    # 在 Control 组里构建最近邻搜索
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(data.loc[control_idx, ['ps_score']])
    distances, indices = nbrs.kneighbors(data.loc[treated_idx, ['ps_score']])

    # 获取匹配到的 Control 样本的索引
    matched_control_idx = control_idx[indices.flatten()]

    # 5. 计算因果效应 (ATT: Average Treatment Effect on the Treated)
    # ATT = Mean(Y_treated - Y_matched_control)
    y_treated = data.loc[treated_idx, outcome_col].values
    y_matched = data.loc[matched_control_idx, outcome_col].values

    att = np.mean(y_treated - y_matched)

    # 计算相对变化的百分比 (更有论文解释力)
    baseline_mean = np.mean(y_matched)
    percent_change = (att / baseline_mean) * 100

    # 6. 显著性检验 (Paired t-test)
    from scipy.stats import ttest_rel
    t_stat, p_val = ttest_rel(y_treated, y_matched)

    print(f"  - ATT (Absolute Effect): {att:.4f}")
    print(f"  - ATT (Relative Change): {percent_change:+.2f}%")
    print(f"  - P-value: {p_val:.4e} {'***' if p_val < 0.001 else '*'}")

    return {
        'treatment': treatment_col,
        'outcome': outcome_col,
        'att_rel': percent_change,
        'p_val': p_val
    }


# ==========================================
# 主程序示例
# ==========================================
if __name__ == "__main__":
    # 模拟数据加载 (请替换为你的真实数据)
    # df_final = pd.read_csv("your_merged_data.csv")
    df_final, attack_cols = load_and_process_data()

    # print(df_final.head(0).to_string())

    # ================= 实验设计 =================

    # 实验 1: 验证 "计算复杂度 (JS Time)" 对 "时间模式 (IAT)" 的因果影响
    # 混杂变量 (需要控制的):
    #   - Total Transferred Bytes (控制大小)
    #   - Rank (控制网站热门程度)
    #   - HTTP3 Ratio (控制协议对时间的天然影响)

    res1 = perform_psm_causal_inference(
        df_final,
        treatment_col='JS Exec Time',  # 干预: JS 执行时间
        outcome_col='Mean_IAT',  # 结果: 平均包间隔 (你需要提取这个特征)
        confounder_cols=['Total Transferred Bytes', 'Rank', 'HTTP3 Ratio']  # 控制变量
    )

    # 实验 2: 验证 "传输体积 (Bytes)" 对 "突发结构 (Burst Size)" 的因果影响
    # 混杂变量 (需要控制的):
    #   - JS Exec Time (控制计算延迟)
    #   - Rank

    res2 = perform_psm_causal_inference(
        df_final,
        treatment_col='Total Transferred Bytes',  # 干预: 总字节数
        outcome_col='Mean_Burst_Size',  # 结果: 平均突发大小 (你需要提取这个特征)
        confounder_cols=['JS Exec Time', 'Rank']  # 控制变量
    )

    res3 = perform_psm_causal_inference(
        df_final,
        treatment_col='JS Exec Time',  # 干预: 总字节数
        outcome_col='Burst_Count',  # 结果: 平均突发大小 (你需要提取这个特征)
        confounder_cols=['Total Transferred Bytes', 'Rank']  # 控制变量
    )

    res4 = perform_psm_causal_inference(
        df_final,
        treatment_col='JS Exec Time',  # 干预: 总字节数
        outcome_col='Std_IAT',  # 结果: 平均突发大小 (你需要提取这个特征)
        confounder_cols=['Total Transferred Bytes', 'Rank']  # 控制变量
    )

    res5 = perform_psm_causal_inference(
        df_final,
        treatment_col='HTTP3 Ratio',  # 干预: 总字节数
        outcome_col='Mean_IAT',  # 结果: 平均突发大小 (你需要提取这个特征)
        confounder_cols=['Total Transferred Bytes', 'JS Exec Time']  # 控制变量
    )

    # print("\n[Conclusion]")
    # print(
    #     f"1. High JS Complexity causes {res1['att_rel']:.1f}% increase in Packet Delay (IAT), independent of site size.")
    # print(
    #     f"2. High Traffic Volume causes {res2['att_rel']:.1f}% increase in Burst Size, independent of rendering logic.")
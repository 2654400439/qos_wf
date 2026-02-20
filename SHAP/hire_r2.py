import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error  # 引入 MAE


# ... (在此处保持 load_and_process_data 函数不变) ...
# ... (在此处保持 run_hierarchical_analysis 函数不变，但稍微修改返回值以包含 MAE) ...

# 为了方便，我把微调后的 run_hierarchical_analysis 和 main 写在这里
# 请将下面的代码替换你原脚本的后半部分

# ==========================================
# 1. 数据加载 (保持你提供的版本不变)
# ==========================================
def load_and_process_data():
    print("\n--- 开始加载数据 ---")
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
    attack_cols = [col for col in file_map.keys() if col in df_f1.columns]
    df_f1['F1_Mean'] = df_f1[attack_cols].mean(axis=1)
    df_f1.dropna(subset=['F1_Mean'], inplace=True)

    df_qos = pd.read_csv("../preprocessed_data/domain_metrics_full.csv")
    df_qos['Domain'] = df_qos['Domain'].astype(str).str.strip()

    df_final = pd.merge(df_f1, df_qos, on="Domain", how="inner")

    feature_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()
    targets_to_exclude = attack_cols + ['F1_Mean', 'F1_Max']
    for target in targets_to_exclude:
        if target in feature_cols: feature_cols.remove(target)
    if 'TCP Handshake' in feature_cols: feature_cols.remove('TCP Handshake')

    for col in feature_cols:
        if df_final[col].isnull().any():
            df_final[col] = df_final[col].fillna(df_final[col].median())

    return df_final, attack_cols


# ==========================================
# 2. 分层回归分析 (微调：增加 MAE 输出)
# ==========================================
def run_hierarchical_analysis(df, feature_groups, target_col):
    y = df[target_col]
    valid_indices = y.dropna().index
    y = y.loc[valid_indices]
    df_subset = df.loc[valid_indices]

    results = []
    current_features = []
    prev_r2 = 0

    for group_name, cols in feature_groups.items():
        valid_cols = [c for c in cols if c in df_subset.columns]
        current_features.extend(valid_cols)

        if not current_features:
            continue

        X = df_subset[current_features]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)  # 计算平均绝对误差

        delta_r2 = r2 - prev_r2
        prev_r2 = r2

        top_driver = "-"
        if len(current_features) > 0:
            imps = model.feature_importances_
            top_idx = imps.argmax()
            # 缩写一下特征名以便表格好看
            raw_driver = current_features[top_idx]
            top_driver = raw_driver.replace("Total Transferred Bytes", "Bytes") \
                .replace("JS Exec Time", "JS_Time") \
                .replace("HTTP3 Ratio", "HTTP3")

        results.append({
            'Target': target_col,
            'Stage': group_name,
            'R2': r2,
            'R2_Inc': delta_r2,
            'MAE': mae,
            'Top_Driver': top_driver
        })

    return results


# ==========================================
# 3. 结果格式化 (生成宽表)
# ==========================================
def format_wide_table(all_results, target_order):
    # 准备数据容器
    table_data = {}

    for row in all_results:
        tgt = row['Target']
        stage = row['Stage']

        if tgt not in table_data:
            table_data[tgt] = {}

        # 格式化单元格内容: "数值 (主要特征)"
        if stage == "Network QoS":
            val = f"{row['R2']:.3f} ({row['Top_Driver']})"
            table_data[tgt]['1. Network QoS (R2)'] = val

        elif stage == "Runtime & QoE":
            # 对于增量，如果是正数加号，负数保持
            inc_sign = "+" if row['R2_Inc'] >= 0 else ""
            val = f"{inc_sign}{row['R2_Inc']:.3f} ({row['Top_Driver']})"
            table_data[tgt]['2. + Runtime (Inc)'] = val

            # 最后一层时，记录总分
            table_data[tgt]['3. Final R2'] = f"{row['R2']:.3f}"
            table_data[tgt]['4. Final MAE'] = f"{row['MAE']:.3f}"

    # 转为 DataFrame
    df_wide = pd.DataFrame(table_data)

    # 调整列顺序: F1_Mean 第一，其他按 target_order 排
    cols = ['F1_Mean'] + [c for c in target_order if c in df_wide.columns and c != 'F1_Mean']
    df_wide = df_wide[cols]

    # 调整行顺序
    row_order = ['1. Network QoS (R2)', '2. + Runtime (Inc)', '3. Final R2', '4. Final MAE']
    df_wide = df_wide.reindex(row_order)

    return df_wide


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    df_final, attack_cols = load_and_process_data()

    # 定义特征组
    feature_groups = {
        "Network QoS": ['TLS Setup', 'HTTP3 Ratio', 'Total Transferred Bytes', 'DNS Lookup Time'],
        "Runtime & QoE": ['JS Exec Time', 'Paint Count', 'Layout Count', 'TTI', 'LCP']
    }

    all_results = []
    target_list = ['F1_Mean'] + [c for c in attack_cols if c in df_final.columns]

    if df_final is not None and not df_final.empty:
        print(f"\nProcessing Targets: {target_list} ...")

        for target in target_list:
            res = run_hierarchical_analysis(df_final, feature_groups, target_col=target)
            all_results.extend(res)

        # 生成最终大表
        df_summary = format_wide_table(all_results, attack_cols)

        print("\n" + "=" * 100)
        print("Final Hierarchical Regression Summary (High Density)")
        print("Format: Score (Top Driver Feature)")
        print("=" * 100)
        print(df_summary)

        df_summary.to_csv("hierarchical_summary_matrix.csv")
        print("\nResult saved to 'hierarchical_summary_matrix.csv'")
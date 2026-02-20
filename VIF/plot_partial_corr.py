import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 准备数据 (基于你提供的真实输出)
data = {
    "Metric": [
        "Load Time", "JS Exec Time", "JS Long Task Count",
        "Paint Count", "Style Recalc", "FCP",
        "Layout Count", "Speed Index", "LCP", "TCP Handshake"
    ],
    "Raw Correlation": [
        0.175783, 0.158382, 0.151141,
        0.144043, 0.143617, 0.137471,
        0.137538, 0.132892, 0.114933, 0.107484
    ],
    "Partial Correlation": [
        0.163799, 0.146614, 0.135887,
        0.125168, 0.125308, 0.124094,
        0.117098, 0.112453, 0.093587, 0.103314
    ]
}

df = pd.DataFrame(data)

# 按 Raw Correlation 排序
df = df.sort_values("Raw Correlation", ascending=True)

# 2. 绘图设置
plt.figure(figsize=(8, 6))
plt.rcParams['font.family'] = 'sans-serif' # 论文常用字体
plt.rcParams['font.size'] = 11

# 设置位置
y = np.arange(len(df))
height = 0.35  # 条形高度

# 3. 绘制条形图
# 绘制 "Raw Correlation" (作为背景/总影响)
plt.barh(y + height/2, df["Raw Correlation"], height, label='Raw Correlation',
         color='#d9d9d9', edgecolor='none') # 浅灰色

# 绘制 "Partial Correlation" (作为核心/净影响)
plt.barh(y - height/2, df["Partial Correlation"], height, label='Partial Correlation (Size-Controlled)',
         color='#2c7bb6', edgecolor='none') # 深蓝色

# 4. 美化图表
plt.xlabel("Spearman Correlation Coefficient ($\\rho$)", fontsize=12, fontweight='bold')
plt.yticks(y, df["Metric"])
plt.title("Robustness Check: QoS Metrics vs. Website Size", fontsize=13, pad=15)

# 添加图例
plt.legend(loc='lower right', frameon=True)

# 添加网格线
plt.grid(axis='x', linestyle='--', alpha=0.5)

# 在条形图旁标注保留率 (Retention Rate)
for i, (raw, partial) in enumerate(zip(df["Raw Correlation"], df["Partial Correlation"])):
    retention = (partial / raw) * 100
    plt.text(raw + 0.005, i + height/2, f"{retention:.1f}% Retained",
             va='center', fontsize=9, color='#555555')

plt.tight_layout()

# 5. 保存
plt.savefig("partial_correlation_analysis.pdf", bbox_inches='tight')
plt.show()
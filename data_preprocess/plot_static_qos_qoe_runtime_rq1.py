import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from qos_qoe_runtime_mediam_store import main_result


top, torso, tail, runtime_top, runtime_torso, runtime_tail, qoe_top, qoe_torso, qoe_tail = main_result()

data = []
for k, v in top.items():
    tail_v = tail[k]
    difference = round((v - tail_v) / tail_v * 100, 1)
    if k in ['HTTP3 Ratio', 'CDN Ratio', 'TLS 1.3 Ratio']:
        data.append((k+'*', "Web QoS", difference))
    elif k == 'Connection Reuse Ratio':
        data.append(('Conn. Reuse Ratio*', "Web QoS", difference))
    else:
        data.append((k, "Web QoS", difference))

for k, v in runtime_top.items():
    tail_v = runtime_tail[k]
    difference = round((v - tail_v) / tail_v * 100, 1)
    data.append((k, "Runtime", difference))

for k, v in qoe_top.items():
    tail_v = qoe_tail[k]
    difference = round((v - tail_v) / tail_v * 100, 1)
    data.append((k, "QoE", difference))

df = pd.DataFrame(data, columns=["Metric", "Category", "Diff"])

# --- 2. 核心数据处理 ---

# 定义分类顺序
category_order = ["Web QoS", "Runtime", "QoE"]
df['Category'] = pd.Categorical(df['Category'], categories=category_order, ordered=True)

# 组内排序：数值小的在上面 (Visual Top)，数值大的在下面
# 注意：因为最后我们要 invert_yaxis，所以这里应该按升序排
df = df.sort_values(by=['Category', 'Diff'], ascending=[True, True])

# --- 3. 绘图配置 (论文级样式) ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12  # 全局基础字号

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# 截断阈值
CUTOFF_VAL = 65
X_LIMIT = (-85, 85)  # 控制画布左右留白

# 计算 Y 轴位置 (手动布局以插入标题)
y_pos_map = {}
labels_map = {}
current_y = 0
header_positions = []  # 存储每组标题的位置

# 反向遍历 Category，因为我们要从上往下画 (对应 Y 轴从大到小或者 invert)
# 这里我们可以简单点：Y=0 是最上面，递增是往下。
# 为了方便 matplotlib 的 barh (默认 Y 向上)，我们倒着算，或者画完 invert。
# 这里采用：画完 invert。所以先处理第一个组 "Web QoS"

plot_data = []  # 存储 (y, value, color, hatch, original_value)

for cat in category_order:
    sub_df = df[df['Category'] == cat]

    # 1. 记录标题位置 (在当前组所有数据的上方留空)
    # 我们给标题留 1.5 个单位的高度
    header_y = current_y
    header_positions.append((cat, header_y))
    current_y += 1.2  # 标题占位

    # 2. 遍历组内数据
    for idx, row in sub_df.iterrows():
        val = row['Diff']

        # 截断逻辑
        is_truncated = False
        plot_val = val
        if val > CUTOFF_VAL:
            plot_val = CUTOFF_VAL
            is_truncated = True
        elif val < -CUTOFF_VAL:
            plot_val = -CUTOFF_VAL
            is_truncated = True

        color = '#d62728' if val > 0 else '#2ca02c'  # 红/绿
        hatch = '////' if is_truncated else None

        plot_data.append({
            'y': current_y,
            'val': plot_val,
            'color': color,
            'hatch': hatch,
            'orig_val': val,
            'label': row['Metric']
        })
        current_y += 1  # 下一行

    current_y += 0.8  # 组间距

# --- 4. 开始绘图 ---
fig, ax = plt.subplots(figsize=(8.5, 11))  # 保持高长型，适配论文单栏或整页

# 解包数据
ys = [d['y'] for d in plot_data]
vals = [d['val'] for d in plot_data]
colors = [d['color'] for d in plot_data]
hatches = [d['hatch'] for d in plot_data]

# 绘制柱状图
bars = ax.barh(ys, vals, color=colors, height=0.65, alpha=0.85, edgecolor='none')

# 应用 Hatch (针对截断数据)
for bar, hatch in zip(bars, hatches):
    if hatch:
        bar.set_hatch(hatch)
        bar.set_edgecolor('white')
        bar.set_linewidth(0)

# 绘制 Y 轴文本标签 (Metric Name)
# 我们不使用 set_yticks，而是直接 text，这样可以控制对齐
ax.set_yticks([])  # 隐藏默认 Y 轴
ax.set_yticklabels([])

# 添加左侧 Metric 标签
for d in plot_data:
    # 放在 Y 轴左侧一点点
    ax.text(X_LIMIT[0], d['y'], d['label'],
            ha='right', va='center', fontsize=11, color='#333')

# --- 5. 添加组标题 (Section Headers) ---
# 这是解决重叠的关键：把标题画在图里
for cat, y in header_positions:
    # 画一条淡淡的横线作为分隔 (可选，看你喜好)
    ax.axhline(y + 0.3, xmin=0, xmax=1, color='gray', linewidth=0.5, alpha=0.3, linestyle='--')

    # 标题文字：居中或者居左。
    # 这里建议居左对齐到 Metric 标签的上方，或者居中
    # 居中显示看起来比较大气
    ax.text(0, y + 0.45, f"——  {cat}  ——",
            ha='center', va='center', fontsize=12, fontweight='bold',
            color='black', backgroundcolor='white')

# --- 6. 数值标签与装饰 ---
ax.axvline(0, color='black', linewidth=0.8)

for d in plot_data:
    v = d['orig_val']
    plotted_v = d['val']
    y = d['y']

    # 标签位置
    offset = 2
    ha = 'left' if v > 0 else 'right'
    text_x = plotted_v + (offset if v > 0 else -offset)

    # 格式化数值
    label_text = f"+{v:.1f}%" if v > 0 else f"{v:.1f}%"

    # 绘制
    # 特殊处理 TBT 这种超大值，字体可以稍微醒目一点
    font_weight = 'bold' if abs(v) > 100 else 'normal'

    ax.text(text_x, y, label_text, va='center', ha=ha,
            fontsize=10, fontweight=font_weight, color='black')

# 设置 X 轴
ax.set_xlim(X_LIMIT)
ax.set_xlabel('Relative Difference (%)', fontsize=12, fontweight='bold')

# 顶部注解 (优化排版)
ax.set_title('Optimization Gap: Top 500 vs. Tail Websites', fontsize=16, pad=35)
ax.text(0.25, 1.015, "← Top 500 is Optimized / Faster", transform=ax.transAxes,
        color='#2ca02c', fontsize=11, fontweight='bold', ha='center')
ax.text(0.75, 1.015, "Top 500 is Complex / Heavier →", transform=ax.transAxes,
        color='#d62728', fontsize=11, fontweight='bold', ha='center')

# 网格
ax.grid(axis='x', linestyle='--', alpha=0.3)

# 反转 Y 轴 (因为我们是从 0 开始累加 y 的，0 在最下面，invert 后 0 在最上面)
ax.invert_yaxis()
# 去除多余边框
sns.despine(left=True, bottom=False)

plt.tight_layout()
plt.savefig('qos_top_tail.pdf', bbox_inches='tight', dpi=300)
plt.show()
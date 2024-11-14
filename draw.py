import matplotlib.pyplot as plt
import numpy as np

# 设置绘图的字体
plt.rc('font', family='serif', size=12)
plt.rc('axes', titlesize=14)     # 轴标题
plt.rc('axes', labelsize=12)     # 轴标签
plt.rc('legend', fontsize=10)     # 图例字体
plt.rc('lines', linewidth=2)      # 线条宽度
plt.rc('lines', markersize=6)     # 标记大小

# 示例数据
k_values = np.arange(1, 18)  # K值为1到17

# 创建折线图
plt.figure(figsize=(12, 8))

# 颜色和标记列表
colors = plt.cm.viridis(np.linspace(0, 1, 17))  # 生成17种颜色
markers = ['o', 's', '^', 'D', 'p', '*', 'x', 'h', 'v', '<', '>', '1', '2', '3', '4', '8']

# 生成并绘制17条曲线
for i in range(17):
    score_values = np.random.randint(5, 50, size=len(k_values))  # 随机生成得分数据
    plt.plot(k_values, score_values, marker=markers[i % len(markers)], linestyle='-', 
             color=colors[i], alpha=0.8, label=f'Experiment {i+1}')

# 添加标题和标签
plt.title('Score vs K for Multiple Experiments', fontsize=16)
plt.xlabel('K', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.xticks(k_values)  # 设置x轴刻度
plt.grid(True, linestyle='--', alpha=0.7)

# 显示图例
plt.legend(ncol=2, bbox_to_anchor=(1.05, 1), loc='upper left')  # 设置图例位置

# 保存图片为PDF格式
plt.savefig('multiple_experiments_results.pdf', bbox_inches='tight', format='pdf')

# 展示图形
plt.show()
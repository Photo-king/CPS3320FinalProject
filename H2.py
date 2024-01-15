import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

# 读取CSV文件
df = pd.read_csv('us-counties.csv')

# 将日期列转换为datetime类型，并按日期排序
df['date'] = pd.to_datetime(df['date'])
df.sort_values(['state', 'date'], inplace=True)

# 使用pivot_table函数，将每个州作为列，日期作为行，确诊数作为值
df_pivot = pd.pivot_table(df, values='cases', index='date', columns='state')

# 绘制不同state的cases数量变化图
fig, ax = plt.subplots(figsize=(20, 13))

# 获取所有state的列表
states = df['state'].unique()

# 循环绘制每个州的线条和拟合曲线
for state in states:
    y = df_pivot[state]
    if not np.isnan(y).any():
        x = (y.index - y.index[0]).days
        coefs = np.polyfit(x, y, deg=6)
        y_fit = np.polyval(coefs, x)
        ax.plot(y.index, y, label=state)
        ax.plot(y.index, y_fit, '--', label=state + ' (fit)')

# 设置纵坐标的格式化器（不显示科学计数法）
formatter = ScalarFormatter(useOffset=False)
formatter.set_scientific(False)
ax.yaxis.set_major_formatter(formatter)

# 设置图形的标题、标签和图例
ax.set_title('Cases by State')
ax.set_xlabel('Date')
ax.set_ylabel('Cases')
ax.legend()

plt.show()

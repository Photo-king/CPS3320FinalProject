import pandas as pd
import matplotlib
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')#不看警告！

# GPT老师说，这样不会报错
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# 读取CSV文件
df = pd.read_csv('Ame.csv')

# 将日期列转换为datetime类型，并按日期排序
df['date'] = pd.to_datetime(df['date'])

# 使用pivot_table函数，将每个州作为列，日期作为行，确诊数作为值
df_pivot = df.pivot_table(values='cases', index='date', columns='state')
df_pivot.fillna(0, inplace=True)#避免误删除2020年数据
df_pivot.dropna(axis=0, how='any', inplace=True)  # 大拇指！

# 绘制不同state的cases数量变化图
fig, ax = plt.subplots(figsize=(20, 13))

# 获取所有state的列表
states = df['state'].unique()

# 循环绘制每个州的线条和拟合曲线
actual_lines = []  # 存储所有州的实际曲线
fit_lines = []  # 存储所有州的拟合曲线
forecast_lines = []  # 存储所有州的预测曲线
# 模型参数
order = (4, 2, 0)  # 这是ARIMA模型的参数，需要根据实际数据调整

for state in states:
    y = df_pivot[state]
    x = (y.index - y.index[0]).days.values
    y_num = pd.to_numeric(y, errors='coerce')
    coefs = np.polyfit(x, y_num, deg=9)#6
    y_fit = np.polyval(coefs, x)
    model = ARIMA(y_num, order=order)
    model_fit = model.fit()
    # 预测未来90天
    forecast = model_fit.get_forecast(steps=90)
    y_forecast = forecast.predicted_mean
    # 合并原始数据和预测数据
    y_total = y_forecast
    fit_lines.append((y.index, y_fit, state + ' (fit)'))
    actual_lines.append((y.index, y, state))
    forecast_lines.append((y_total.index, y_total, state + ' (forecast)'))
# 一次性绘制所有曲线
for line in actual_lines:
    ax.plot(line[0], line[1], label=line[2])
for line in fit_lines:
    ax.plot(line[0], line[1], '--', label=line[2])
for line in forecast_lines:
    ax.plot(line[0], line[1], ':', label=line[2])

#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# # 绘制ACF图
# plot_acf(y)
# plt.show()
#
# # 绘制PACF图
# plot_pacf(y)
# plt.show()

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
print("Finish")

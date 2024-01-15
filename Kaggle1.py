import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 读取数据
data = pd.read_csv('us-counties.csv')

# 对日期进行处理
data['date'] = pd.to_datetime(data['date'])

# 根据需要选择国家或州进行分析
county_data = data[data['county'] == 'Columbia']
country_data = county_data.groupby('date').sum().reset_index()  # 按日期进行聚合

# 创建预测模型
X = np.array(range(len(country_data))).reshape(-1, 1)
y = country_data['cases']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 计算预测误差
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# 绘制预测图
plt.figure(figsize=(12, 6))
plt.plot(country_data['date'], country_data['cases'], label='Actual cases')
plt.plot(country_data['date'][len(X_train):], predictions, label='Predicted cases')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.title('COVID-19 Cases Prediction')
plt.legend()
plt.show()

# 预测未来走势
future_days = 30  # 预测未来30天
future_dates = pd.date_range(country_data['date'].iloc[-1] + pd.Timedelta(days=1), periods=future_days, freq='D')
future_X = np.array(range(len(country_data), len(country_data) + future_days)).reshape(-1, 1)
future_predictions = model.predict(future_X)

plt.figure(figsize=(12, 6))
plt.plot(country_data['date'], country_data['cases'], label='Actual cases')
plt.plot(future_dates, future_predictions, label='Future Predicted cases')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.title('COVID-19 Future Cases Prediction')
plt.legend()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from fbprophet import Prophet

# 读取数据
data = pd.read_csv("us-counties.csv")

# 数据预处理
data["date"] = pd.to_datetime(data["date"])
data = data.groupby("date")[["cases", "deaths"]].sum().reset_index()

# 绘制累计确诊病例和死亡病例趋势图
plt.figure(figsize=(12, 6))
plt.plot(data["date"], data["cases"], label="Cases")
plt.plot(data["date"], data["deaths"], label="Deaths")
plt.xlabel("Date")
plt.ylabel("Number")
plt.title("Covid19 Cases and Deaths in US (2020-2021)")
plt.legend()
plt.show()

# 数据分析与预测
# 以确诊病例为例
cases = data[["date", "cases"]].rename(columns={"date": "ds", "cases": "y"})

# 使用Prophet模型预测
model = Prophet()
model.fit(cases)
future = model.make_future_dataframe(periods=30)  # 预测未来30天
forecast = model.predict(future)

# 绘制预测结果
fig = model.plot(forecast)
plt.title("Covid19 Cases Forecast in US (Prophet)")
plt.show()

# 使用ARIMA模型预测
arima_model = auto_arima(cases["y"], seasonal=True, m=7)
arima_forecast = arima_model.predict(n_periods=30)  # 预测未来30天

# 绘制预测结果
plt.figure(figsize=(12, 6))
plt.plot(cases["ds"], cases["y"], label="Observed")
plt.plot(pd.date_range(cases["ds"].iloc[-1], periods=31)[1:], arima_forecast, label="Forecast (ARIMA)")
plt.xlabel("Date")
plt.ylabel("Cases")
plt.title("Covid19 Cases Forecast in US (ARIMA)")
plt.legend()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import ssl

# 禁用SSL证书验证（有安全风险，请谨慎使用）
ssl._create_default_https_context = ssl._create_unverified_context

# 加载数据
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
df = pd.read_csv(url)

# 数据预处理
df = df.drop(['Lat', 'Long'], axis=1)
df = df.groupby('Country/Region').sum()
df = df.T

# 绘制累计确诊人数折线图
df[['China', 'US', 'India', 'Brazil']].plot()
plt.show()

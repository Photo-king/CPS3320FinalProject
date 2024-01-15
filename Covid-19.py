import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from statsmodels.tsa.arima.model import ARIMA
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import warnings
warnings.filterwarnings('ignore')

def data_pre_process():
    df = pd.read_csv('Ame.csv')#读取文件
    #预处理数据：将日期列转换为datetime类型，并按日期排序
    df['date'] = pd.to_datetime(df['date'])
    # 使用pivot_table函数，将每个州作为列，日期作为行，确诊数作为值
    df_pivot = df.pivot_table(values='cases', index='date', columns='state')
    df_pivot.fillna(0, inplace=True)  # 避免误删除2020年数据
    df_pivot.dropna(axis=0, how='any', inplace=True)  # 代替掉不要的数据
    # 绘制不同state的cases数量变化图大小
    fig, ax = plt.subplots(figsize=(20, 13))
    # 获取所有state的列表
    states = df['state'].unique()
    # 创建字典保存线条数据
    actual_lines_dict = {}
    #存入数据到字典当中
    for state in states:
        y = df_pivot[state]
        x = (y.index - y.index[0]).days.values
        # 合并原始数据
        actual_lines.append((y.index, y, state))
        # 保存数据到字典
        actual_lines_dict[state] = (y.index, y)
    return state

def data_predict():
    # 模型参数
    order = (4, 2, 0)  # 这是ARIMA模型的参数，需要根据实际数据调整
    for state in data_pre_process():
        forecast_lines_dict = {}
        model = ARIMA(y_num, order=order)
        y_num = pd.to_numeric(y, errors='coerce')
        model_fit = model.fit()
        # 预测未来90天
        forecast = model_fit.get_forecast(steps=90)
        y_forecast = forecast.predicted_mean
        y_total = y_forecast
        forecast_lines.append((y_total.index, y_total, state + ' (forecast)'))  # 无用
        forecast_lines_dict[state] = (y_total.index, y_total)



if __name__ == '__main__':
    # 循环绘制每个州的线条和拟合曲线
    actual_lines = []  # 存储所有州的实际曲线
    fit_lines = []  # 存储所有州的拟合曲线
    forecast_lines = []  # 存储所有州的预测曲线
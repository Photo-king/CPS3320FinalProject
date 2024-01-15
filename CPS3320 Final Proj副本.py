import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.ticker import ScalarFormatter
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import datetime

warnings.filterwarnings('ignore')  # Ignore warnings!


def load_data():
    df = pd.read_csv('Amep.csv')
    df['date'] = pd.to_datetime(df['date'])
    df_pivot = df.pivot_table(values='cases', index='date', columns='state')
    df_pivot.fillna(0, inplace=True)
    df_pivot.dropna(axis=0, how='any', inplace=True)
    return df_pivot, df['state'].unique()


def scale_data(data):
    data_df = pd.DataFrame(data)  # Convert data to Pandas DataFrame
    data_df = data_df.dropna()  # Remove rows or columns containing NaN values
    if data_df.shape[0] < 1:  # Check if there is at least one sample in the dataset
        raise ValueError("Data has no samples after removing NaN values.")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_df)
    return scaler, scaled_data


def unscale_data(scaler, scaled_data):
    unscaled_data = scaler.inverse_transform(scaled_data)
    return unscaled_data


def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=64, activation='relu', input_shape=input_shape))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')
    return model


def create_gui():
    df_pivot, states = load_data()#前面的获取数据
    actual_lines_dict, forecast_lines_dict = forecast(df_pivot, states)#获瞄好点的数据到字典，准备在后面传过去输出
    fig, ax = plt.subplots(figsize=(20, 13))  # fig是figure对象,包含多个图,Axes对象就可以理解为一个图片。
    root = tk.Tk()  # Main window(blank board)
    root.title("State Selection")#添加标题
    for i, state in enumerate(states):                 #函数命令    获取参数s=state  调用plot_state_data 传入参数（）
        button = tk.Button(root, text=state, command=lambda s=state: plot_state_data(s, ax, fig, actual_lines_dict,forecast_lines_dict, root))
        button.grid(row=1 + (i // 2), column=i % 2)
    formatter = ScalarFormatter(useOffset=False)#？？
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_title('Cases by State')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cases')
    ax.legend()
    print("Finish")
    root.mainloop()#start main loop in tkinter~


def forecast(df_pivot, states):
    actual_lines_dict = {}  # Use dict to store two type lines to help to output the lines
    forecast_lines_dict = {}
    for state in states:
        #y = the cases of the specific state and x = day offset
        y = df_pivot[state]
        x = (y.index - y.index[0]).days.values
        y_num = pd.to_numeric(y, errors='coerce')#Turn not number character into NaN

        # Filter data based on date range
        start_date = pd.to_datetime('2021-08-01')  # Determine the range for training
        end_date = pd.to_datetime('2021-10-15')
        mask = (y.index >= start_date) & (y.index < end_date)#是否符合时间要求？
        y_filtered = y[mask]#yes or not
        y_num_filtered = y_num[mask]#在指定日期范围内，选定州的病例数据的数值表示，可以确保在进行数据处理和模型训练时，只使用了符合日期范围条件的数值型病例数据。

        # Scale the data
        scaler, scaled_data = scale_data(y_num_filtered.values.reshape(-1, 1))#p1一维参数p2二位参数m1使其范围限定在一个特定的区间内 增加训练结果和准确性
#我们首先将数据变换维度，然后处理，然后放回，就像编码之后方便计算，保存，在需要使用的时候再解码
        # Prepare training data
        x_train = []
        y_train = []#初始化空的列表，用于存储训练数据。
        seq_length = 30  #设置序列的长度，即每个样本的时间步数。在这个例子中，选择了一个长度为30的序列。
        for i in range(seq_length, len(scaled_data)):#循环迭代从 seq_length 开始到 scaled_data 的长度减1的范围，以构建训练样本。
            x_train.append(scaled_data[i - seq_length:i, 0])
            y_train.append(scaled_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)#将列表转换为 NumPy 数组，以便后续的处理和模型训练。
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))#重塑为 LSTM 模型所需的三维形状，即 (样本数, 时间步数, 特征数)。在这个例子中，特征数为 1。

        # Create and train LSTM model         #通过调用 create_lstm_model() 函数，我们创建了一个配置了相应层和参数的 LSTM 模型。
        model = create_lstm_model((x_train.shape[1], 1))#loss:监督损失值，5连续时间块没有改善，就终止 verbose=0(不输出训练的详情)
        early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=0)  #用于在训练过程中根据训练损失值的变化情况来提前停止训练。
        model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)#使用训练数据 x_train 和目标标签 y_train
        # 来训练 LSTM 模型。epochs=100 表示训练的迭代次数，batch_size=32 表示每次迭代中的样本批次大小。callbacks=[early_stopping]
        # 表示使用之前定义的 EarlyStopping 回调函数。verbose=0 表示不输出训练过程中的详细信息。

        # Prepare forecasting data
        last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1)
        forecast_data = []
        num_forecasts = 15  # Forecast for 15 days
        for _ in range(num_forecasts):#通过循环迭代进行预测，每次预测一个时间步。这是一个时间预测
            next_pred = model.predict(last_sequence)[0, 0]#使用训练好的 LSTM 模型对最后一个序列进行预测，得到下一个时间步的预测结果。
            # ↑返回预测结果的数组，其中 [0, 0] 表示取出预测结果的第一个元素。
            forecast_data.append(next_pred)#将预测结果添加到 forecast_data 列表中，存储所有时间步的预测结果。
            last_sequence = np.roll(last_sequence, -1)#进行循环移位操作，将序列向前移动一个时间步。
            last_sequence[0, -1, 0] = next_pred#将最后一个时间步的值更新为当前预测结果，以便作为下一个时间步的输入。

        # Unscale the data# 归一化的预测结果恢复为原始数据的形式。
        unscaled_forecast = unscale_data(scaler, np.array(forecast_data[:num_forecasts]).reshape(-1, 1))

        # Prepare date index for forecasted values
        last_date = y_filtered.index[-1]#获取经过日期筛选的选定州病例数据 y_filtered 的最后一个日期。
        forecast_dates = pd.date_range(start=last_date + datetime.timedelta(days=1), periods=num_forecasts)
        #根据最后一个日期 last_date，使用 pd.date_range() 函数生成一个日期范围，以用于表示未来预测结果的日期。
        # start=last_date + datetime.timedelta(days=1) 表示从最后一个日期的后一天开始，periods=num_forecasts 表示生成 num_forecasts 个连续的日期。

        # Extend actual data by 15 days
        extended_y_filtered = y_filtered.append(pd.Series([None] * num_forecasts, index=forecast_dates))

        actual_lines_dict[state] = (extended_y_filtered.index, extended_y_filtered)
        forecast_lines_dict[state] = (forecast_dates, unscaled_forecast.flatten())
    for state in states:
        y = df_pivot[state]
        x = (y.index - y.index[0]).days.values
        y_num = pd.to_numeric(y, errors='coerce')

        # Filter data based on date range
        start_date = pd.to_datetime('2021-04-01')  # 标定原始数据范围，方便展示
        end_date = pd.to_datetime('2021-11-15')
        mask = (y.index >= start_date) & (y.index < end_date)
        y_filtered = y[mask]
        y_num_filtered = y_num[mask]

        # Scale the data
        scaler, scaled_data = scale_data(y_num_filtered.values.reshape(-1, 1))

        # Prepare training data
        x_train = []
        y_train = []
        seq_length = 30  # Sequence length for LSTM
        for i in range(seq_length, len(scaled_data)):
            x_train.append(scaled_data[i - seq_length:i, 0])
            y_train.append(scaled_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Create and train LSTM model
        model = create_lstm_model((x_train.shape[1], 1))
        early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1)
        model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)

        # Prepare forecasting data
        last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1)
        forecast_data = []
        num_forecasts = 15  # Forecast for 15 days
        for _ in range(num_forecasts):
            next_pred = model.predict(last_sequence)[0, 0]
            forecast_data.append(next_pred)
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[0, -1, 0] = next_pred

        # Unscale the data
        unscaled_forecast = unscale_data(scaler, np.array(forecast_data[:num_forecasts]).reshape(-1, 1))

        # Prepare date index for forecasted values
        last_date = y_filtered.index[-1]
        forecast_dates = pd.date_range(start=last_date + datetime.timedelta(days=1), periods=num_forecasts)

        # Extend actual data by 15 days
        extended_y_filtered = y_filtered.append(pd.Series([None] * num_forecasts, index=forecast_dates))

        actual_lines_dict[state] = (extended_y_filtered.index, extended_y_filtered)

    return actual_lines_dict, forecast_lines_dict


def plot_state_data(state_name, ax, fig, actual_lines_dict, forecast_lines_dict, parent):
    new_window = create_window(state_name, parent)#创建一个新的窗口，并使用 state_name 作为窗口的标题。parent 参数表示父级窗口，在这个程序中是主窗口。
    canvas = FigureCanvasTkAgg(fig, master=new_window)#创建一个基于给定图形 fig 的 FigureCanvasTkAgg 对象，并将其关联到新创建的窗口 new_window 上。
    # canvas.draw()：绘制图形到画布上。
    canvas.draw()
    #将画布的 Tkinter 小部件添加到窗口中，并设置其位置和填充方式。
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
#工具栏对象，用于在图形界面中显示一组交互式工具。
    toolbar = NavigationToolbar2Tk(canvas, new_window)
    toolbar.update()
    #工具栏的 Tkinter 小部件添加到窗口中，并设置其位置和填充方式。
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    ax.clear()
#检查选定州的实际数据和预测数据是否都存在于对应的数据字典中。
    if state_name in actual_lines_dict and state_name in forecast_lines_dict:
        actual_x, actual_y = actual_lines_dict[state_name]
        forecast_x, forecast_y = forecast_lines_dict[state_name]

        # Plot actual data
        ax.plot(actual_x, actual_y, label=state_name + ' (Actual data from 2021-4-1 to 2022-11-15)')

        # Plot forecast data
        ax.plot(forecast_x, forecast_y, ':', label=state_name + ' (Forecast from 2021-11-01 following 15 days)')

        # Connect the last actual data point to the first forecast data point
        ax.plot([actual_x[-1], forecast_x[0]], [actual_y[-1], forecast_y[0]], '--', color='orange')
        ax.set_title('Cases by State')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cases')
        ax.legend()
        canvas.draw()


def create_window(title, parent):
    new_window = tk.Toplevel(parent)
    new_window.title(title)
    return new_window


if __name__ == '__main__':
    create_gui()

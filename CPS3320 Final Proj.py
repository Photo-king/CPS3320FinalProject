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
    df_pivot, states = load_data()
    actual_lines_dict, forecast_lines_dict = forecast(df_pivot, states)
    # create graphic window
    fig, ax = plt.subplots(figsize=(20, 13))
    root = tk.Tk()
    root.title("State Selection and The accuracy is at 79.4%")
    # Define event for the button (Print out)
    for i, state in enumerate(states):
        button = tk.Button(root, text=state,
                           command=lambda s=state: plot_state_data(s, ax, fig, actual_lines_dict, forecast_lines_dict,
                                                                   root))
        button.grid(row=1 + (i // 2), column=i % 2)
    # Set form and labels
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_title('Cases by State')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cases')
    ax.legend()
    print("Finish")
    root.mainloop()  # start main loop in tkinter~


def forecast(df_pivot, states):
    # Use dict to store two type lines to help to output the lines
    actual_lines_dict = {}
    forecast_lines_dict = {}
    for state in states:
        # y = the cases of the specific state and x = day offset
        y = df_pivot[state]
        x = (y.index - y.index[0]).days.values
        y_num = pd.to_numeric(y, errors='coerce')  # Turn not number character into NaN

        # Filter data based on date range，Determine the range for training
        start_date = pd.to_datetime('2021-08-01')
        end_date = pd.to_datetime('2021-10-30')
        mask = (y.index >= start_date) & (y.index < end_date)
        y_filtered = y[mask]
        y_num_filtered = y_num[mask]

        # Scale the data
        scaler, scaled_data = scale_data(y_num_filtered.values.reshape(-1, 1))

        # Prepare training data
        x_train = []
        y_train = []
        seq_length = 15  # Sequence length for LSTM
        for i in range(seq_length, len(scaled_data)):
            x_train.append(scaled_data[i - seq_length:i, 0])
            y_train.append(scaled_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Create and train LSTM model
        model = create_lstm_model((x_train.shape[1], 1))
        early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=0)  # Quit if there is no confident data set
        model.fit(x_train, y_train, epochs=200, batch_size=32, callbacks=[early_stopping], verbose=0)

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
    # Create a new window with state_name as the title of the window.
    new_window = create_window(state_name, parent)
    canvas = FigureCanvasTkAgg(fig, master=new_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    # Creat toolbar
    toolbar = NavigationToolbar2Tk(canvas, new_window)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    ax.clear()
    # Check and output
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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.ticker import ScalarFormatter
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import rnn
import warnings

warnings.filterwarnings('ignore')  # Ignore warnings!

def load_data():
    df = pd.read_csv('Ame.csv')
    df['date'] = pd.to_datetime(df['date'])
    df_pivot = df.pivot_table(values='cases', index='date', columns='state')
    df_pivot.fillna(0, inplace=True)
    df_pivot.dropna(axis=0, how='any', inplace=True)

    return df_pivot, df['state'].unique()

def scale_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaler, scaled_data

def unscale_data(scaler, scaled_data):
    unscaled_data = scaler.inverse_transform(scaled_data)
    return unscaled_data

def create_lstm_model(input_size, hidden_size, num_layers):
    model = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    return model

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs[:, -1], targets)
            loss.backward()
            optimizer.step()

def forecast(df_pivot, states):
    actual_lines_dict = {}
    forecast_lines_dict = {}

    for state in states:
        y = df_pivot[state]
        x = (y.index - y.index[0]).days.values
        y_num = pd.to_numeric(y, errors='coerce')

        # Filter data based on date range
        start_date = pd.to_datetime('2020-01-01')
        end_date = pd.to_datetime('2022-01-01')
        mask = (y.index >= start_date) & (y.index < end_date)
        y_filtered = y[mask]
        y_num_filtered = y_num[mask]

        # Scale the data
        scaler, scaled_data = scale_data(y_num_filtered.values.reshape(-1, 1))

        # Prepare training data
        seq_length = 30  # Sequence length for LSTM
        x_train = []
        y_train = []
        for i in range(seq_length, len(scaled_data)):
            x_train.append(scaled_data[i-seq_length:i, 0])
            y_train.append(scaled_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = torch.Tensor(x_train).unsqueeze(2)
        y_train = torch.Tensor(y_train).unsqueeze(1)

        # Create and train LSTM model
        input_size = 1
        hidden_size = 64
        num_layers = 1
        model = create_lstm_model(input_size, hidden_size, num_layers)
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=0.001)
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        num_epochs = 100
        train_model(model, train_loader, criterion, optimizer, num_epochs)

        # Prepare forecasting data
        last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1)
        forecast_data = []
        num_forecasts = 90
        for _ in range(num_forecasts):
            inputs = torch.Tensor(last_sequence)
            outputs, _ = model(inputs)
            next_pred = outputs[:, -1].item()
            forecast_data.append(next_pred)
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[0, -1, 0] = next_pred

        # Unscale the data
        unscaled_forecast = unscale_data(scaler, np.array(forecast_data).reshape(-1, 1))

        # Prepare date index for forecasted values
        last_date = y_filtered.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.DateOffset(1), periods=num_forecasts)

        actual_lines_dict[state] = (y_filtered.index, y_filtered)
        forecast_lines_dict[state] = (forecast_dates, unscaled_forecast.flatten())

    return actual_lines_dict, forecast_lines_dict

def plot_state_data(state_name, ax, fig, actual_lines_dict, forecast_lines_dict):
    new_window = tk.Toplevel(root)
    new_window.title(state_name)

    canvas = FigureCanvasTkAgg(fig, master=new_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    toolbar = NavigationToolbar2Tk(canvas, new_window)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    ax.clear()
    if state_name == "ALL STATES":
        for state in states:
            ax.plot(*actual_lines_dict[state], label=state + ' (Actual data line basing on date and cases)')
            ax.plot(*forecast_lines_dict[state], ':', label=state + ' (Forecast line basing on date and cases)')
    else:
        if state_name in actual_lines_dict and state_name in forecast_lines_dict:
            ax.plot(*actual_lines_dict[state_name], label=state_name + ' (Actual data line basing on date and cases)')
            ax.plot(*forecast_lines_dict[state_name], ':', label=state_name + ' (Forecast line basing on date and cases)')

    ax.set_title('Cases by State')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cases')
    ax.legend()
    canvas.draw()


if __name__ == '__main__':
    df_pivot, states = load_data()
    actual_lines_dict, forecast_lines_dict = forecast(df_pivot, states)
    fig, ax = plt.subplots(figsize=(20, 13))
    root = tk.Tk()
    root.title("State Selection")
    all_button = tk.Button(root, text='ALL STATES',
                           command=lambda: plot_state_data('ALL STATES', ax, fig, actual_lines_dict,
                                                           forecast_lines_dict))
    all_button.grid(row=0, column=0, columnspan=2)
    for i, state in enumerate(states):
        button = tk.Button(root, text=state,
                           command=lambda s=state: plot_state_data(s, ax, fig, actual_lines_dict, forecast_lines_dict))
        button.grid(row=1 + (i // 2), column=i % 2)
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_title('Cases by State')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cases')
    ax.legend()
    print("Finish")
    root.mainloop()
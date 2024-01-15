import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from statsmodels.tsa.arima.model import ARIMA
import tkinter as tk
from tkinter import ttk

matplotlib.use('TkAgg')

def plot_state_data(state):
    if state in actual_lines_dict and state in fit_lines_dict and state in forecast_lines_dict:
        fig, ax = plt.subplots(figsize=(20, 13))
        ax.plot(*actual_lines_dict[state], label=state)
        ax.plot(*fit_lines_dict[state], '--', label=state + ' (fit)')
        ax.plot(*forecast_lines_dict[state], ':', label=state + ' (forecast)')

        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)

        ax.set_title('Cases by State')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cases')
        ax.legend()
        plt.show()
    else:
        print(f"No data for state: {state}")

df = pd.read_csv('Ame.csv')
df['date'] = pd.to_datetime(df['date'])
df_pivot = df.pivot_table(values='cases', index='date', columns='state')
df_pivot.fillna(0, inplace=True)
df_pivot.dropna(axis=0, how='any', inplace=True)

states = df['state'].unique()
actual_lines = []
fit_lines = []
forecast_lines = []
order = (4, 2, 0)

actual_lines_dict = {}
fit_lines_dict = {}
forecast_lines_dict = {}
for state in states:
    y = df_pivot[state]
    x = (y.index - y.index[0]).days.values
    y_num = pd.to_numeric(y, errors='coerce')
    coefs = np.polyfit(x, y_num, deg=9)
    y_fit = np.polyval(coefs, x)
    model = ARIMA(y_num, order=order)
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=90)
    y_forecast = forecast.predicted_mean
    y_total = y_forecast
    fit_lines.append((y.index, y_fit, state + ' (fit)'))
    actual_lines.append((y.index, y, state))
    forecast_lines.append((y_total.index, y_total, state + ' (forecast)'))
    actual_lines_dict[state] = (y.index, y)
    fit_lines_dict[state] = (y.index, y_fit)
    forecast_lines_dict[state] = (y_total.index, y_total)

def plot_selected_state(event):
    selected_state = state_combobox.get()
    plot_state_data(selected_state)

root = tk.Tk()
state_combobox = ttk.Combobox(root, values=states)
state_combobox.bind("<<ComboboxSelected>>", plot_selected_state)
state_combobox.pack()
root.mainloop()

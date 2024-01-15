import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from statsmodels.tsa.arima.model import ARIMA
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import warnings

warnings.filterwarnings('ignore')  # 不看警告！
matplotlib.use('TkAgg')


def load_data():
    df = pd.read_csv('Ame.csv')
    df['date'] = pd.to_datetime(df['date'])
    df_pivot = df.pivot_table(values='cases', index='date', columns='state')
    df_pivot.fillna(0, inplace=True)
    df_pivot.dropna(axis=0, how='any', inplace=True)

    return df_pivot, df['state'].unique()


def forecast(df_pivot, states):
    order = (4, 2, 0)  # 这是ARIMA模型的参数，需要根据实际数据调整
    actual_lines_dict = {}
    forecast_lines_dict = {}

    for state in states:
        y = df_pivot[state]
        x = (y.index - y.index[0]).days.values
        y_num = pd.to_numeric(y, errors='coerce')
        coefs = np.polyfit(x, y_num, deg=9)
        model = ARIMA(y_num, order=order)
        model_fit = model.fit()
        forecast = model_fit.get_forecast(steps=90)
        y_forecast = forecast.predicted_mean
        y_total = y_forecast
        actual_lines_dict[state] = (y.index, y)
        forecast_lines_dict[state] = (y_total.index, y_total)

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
            ax.plot(*actual_lines_dict[state], label=state + '(Actual data line basing on date and cases)')
            ax.plot(*forecast_lines_dict[state], ':', label=state + ' (forecast line basing on date and cases)')
    else:
        if state_name in actual_lines_dict and state_name in forecast_lines_dict:
            ax.plot(*actual_lines_dict[state_name], label=state_name + '(Actual data line basing on date and cases)')
            ax.plot(*forecast_lines_dict[state_name], ':', label=state_name + ' (forecast line basing on date and cases)')

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

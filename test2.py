actual_y = {'a': 1, 'b': 2, 'c': 3}
forecast_y = {'x': 10, 'y': 20, 'z': 30}

# 获取 actual_y 字典的最后一个值
last_actual_y_value = list(actual_y.values())[-2]

# 获取 forecast_y 字典的最后一个值
last_forecast_y_value = list(forecast_y.values())[-1]

print("actual_y 最后一个值：", last_actual_y_value)
print("forecast_y 最后一个值：", last_forecast_y_value)
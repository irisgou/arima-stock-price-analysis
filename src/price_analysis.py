import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# display all columns
pd.set_option('display.max_columns', None)

def get_valid_file_path():
    while True:
        file_path = input("Please enter the path to the CSV file: ")
        if os.path.isfile(file_path) and file_path.endswith('.csv'):
            return file_path
        else:
            print("Invalid file path or file is not a CSV. Please try again.")

# file_path = os.path.join(os.path.dirname(__file__),'../data/BAJFINANCE.csv')

file_path = get_valid_file_path()

stock_info = pd.read_csv(file_path)
stock_info.head()

stock_info.set_index('Date', inplace=True)

# plot variable weighted average price
stock_info['VWAP'].plot()
plt.show()

lag_features=['High', 'Low', 'Volume', 'Turnover', 'Trades']
window1 = 3
window2 = 7

# find mean
for feature in lag_features:
    stock_info[feature+'rolling_mean_3'] = stock_info[feature].rolling(window=window1).mean()
    stock_info[feature+'rolling_mean_7'] = stock_info[feature].rolling(window=window2).mean()

# find standard deviation
for feature in lag_features:
    stock_info[feature+'rolling_mean_3'] = stock_info[feature].rolling(window=window1).std()
    stock_info[feature+'rolling_mean_7'] = stock_info[feature].rolling(window=window2).std()

# drop NaN values
stock_info.dropna(inplace=True)

# print(stock_info.head())

ind_features=['Highrolling_mean_3', 'Highrolling_mean_7',
       'Lowrolling_mean_3', 'Lowrolling_mean_7', 'Volumerolling_mean_3',
       'Volumerolling_mean_7', 'Turnoverrolling_mean_3',
       'Turnoverrolling_mean_7', 'Tradesrolling_mean_3',
       'Tradesrolling_mean_7', 'Highrolling_std_3', 'Highrolling_std_7',
       'Lowrolling_std_3', 'Lowrolling_std_7', 'Volumerolling_std_3',
       'Volumerolling_std_7', 'Turnoverrolling_std_3', 'Turnoverrolling_std_7',
       'Tradesrolling_std_3', 'Tradesrolling_std_7']

# use first 1800 rows
training_data=stock_info[0:1800]
test_data=stock_info[1800:]

model = auto_arima(y = training_data['VWAP'] , X = training_data[ind_features], trace=True)

model.fit(training_data['VWAP'],training_data[ind_features])

forecast = model.predict(n_periods=len(test_data), X = test_data[ind_features])

test_data['Forecast_ARIMA'] = forecast.values

test_data[['VWAP','Forecast_ARIMA']].plot(figsize=(14,7))
plt.show()

np.sqrt(mean_squared_error(test_data['VWAP'],test_data['Forecast_ARIMA']))

mean_absolute_error(test_data['VWAP'],test_data['Forecast_ARIMA'])
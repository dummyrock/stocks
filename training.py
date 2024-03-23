import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas
import matplotlib.pyplot as plt
import datetime
import alpaca_trade_api as tradeapi
from dataclasses import dataclass

AMOUNT_OF_DAYS_BACK = 60
COLUMS = ["High","Low"]
    
def sorting_input_data(stock_dataset):
    total_stock_dataset = []
    y_data = []
    index = 0
    for i, row in stock_dataset.iterrows():
        stock_list = []
        if index >= AMOUNT_OF_DAYS_BACK and index != (len(stock_dataset)-1):
            for i in range(AMOUNT_OF_DAYS_BACK):
                stock_info_list = []
                stock_info = stock_dataset.iloc[index-(AMOUNT_OF_DAYS_BACK - i)][COLUMS]
                for stock_number_data in stock_info:
                    stock_info_list.append(stock_number_data)
                stock_list.append(stock_info_list)
            total_stock_dataset.append(stock_list)
            y_stock_info = stock_dataset.iloc[index+1]["Open"]
            y_data.append(y_stock_info)
        index += 1
    return np.array(total_stock_dataset), np.array(y_data)

def model(total_data,y_data):
    
    X_train, X_test, y_train, y_test = train_test_split(total_data, y_data, test_size=0.2, random_state=42)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(AMOUNT_OF_DAYS_BACK, len(COLUMS))),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50000, validation_data=(X_test, y_test))
    
    model.save("stock_prediction_model.keras")

def buy_sell(data):
    total = 0
    total_trades = 0
    for i in range(len(data)):
        number = data[i]
        target = data[i-1]
        target2 = data[i-2]
        if number != data[0] and number != data[1]:
            if number > target and number > target2:
                total += 1
                total_trades += 1
            if number < target:
                total -= 1
                total_trades += 1
    return total, total_trades

def predict_and_plot(test_total_data, test_y_data):
    model = tf.keras.models.load_model("stock_prediction_model.keras")

    predictions = model.predict(test_total_data)
    
    pred, pred_tot_trades = buy_sell(predictions)
    real, real_tot_trades = buy_sell(test_y_data)
    
    print(f"The predictions buy sell is {pred} with {pred_tot_trades} trades")
    print(f"The real buy sell is {real} with {real_tot_trades} trades")
    
    plt.plot(predictions)
    plt.plot(test_y_data)
    plt.xlabel('Time')
    plt.ylabel('Open Price')
    plt.title('Predicted Open Price Over Time')
    plt.show()
    
def main():
    
    today = datetime.date.today()
    
    yesterday = today - datetime.timedelta(days=1)
    
    thirty_days_ago = today - datetime.timedelta(days=30)
    
    formatted_thirty_days_ago_date = thirty_days_ago.strftime("%Y-%m-%d")
    
    formatted_yesterday_date = yesterday.strftime("%Y-%m-%d")
    
    formatted_today_date = today.strftime("%Y-%m-%d")
    
    
    #Uncomment these lines to retrain the model
    #train_apple_data = yf.download("AAPL",start=formatted_thirty_days_ago_date, end=formatted_today_date, auto_adjust=True, interval="2m")
    #total_data, y_data = sorting_input_data(train_apple_data)
    #model(total_data,y_data)
    
    #for weekend use or testing comment this line
    test_apple_data = yf.download("AAPL",start=formatted_yesterday_date, end=formatted_today_date, auto_adjust=True, interval="2m")
    
    #use this line for testing or on the weekends where you have to manually enter the date of the friday
    #test_apple_data = yf.download("AAPL",start="2024-03-11", end="2024-03-12", auto_adjust=True, interval="2m")
    
    test_total_data, test_y_data = sorting_input_data(test_apple_data)
    predict_and_plot(test_total_data, test_y_data)

if __name__ == "__main__":
    main()
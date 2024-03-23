import yfinance as yf
import numpy as np
import tensorflow as tf
import pandas
import requests
import datetime
import alpaca_trade_api as tradeapi
import training 
from dataclasses import dataclass

API_KEY = 'cno8r09r01qu79g549mgcno8r09r01qu79g549n0'
TIME_INTERVAL = 2

@dataclass
class Currency:
    capital:float
    stocks_held:int
    price_bought_at:float
    
def buy_sell(data,capital):
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_price = get_current_stock_price()
    print(current_price)
    for i in range(len(data)):
        number = data[i]
        target = data[i-1]
        target2 = data[i-2]
        if number != data[0] and number != data[1]:
            if number > target and number < target2:
                if number == data[len(data)-1]:
                    capital.capital -= current_price
                    capital.stocks_held += 1
                    capital.price_bought_at = current_price
                    print(f"buy at: ${current_price} at the time {current_time}")
            if number < target and capital.stocks_held > 0 and capital.price_bought_at < current_price:
                if number == data[len(data)-1]:
                    for stock in range(capital.stocks_held):
                        capital.capital += current_price - ((current_price - capital.price_bought_at)*0.1)
                        capital.stocks_held -= 1
                    print(f"sell at: ${current_price} at the time {current_time}")
    
def predict(test_total_data,capital):
    model = tf.keras.models.load_model("stock_prediction_model.keras")
    predictions = model.predict(test_total_data)
    buy_sell(predictions,capital)
    
def running(start_time,interval,capital):
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    
    formatted_yesterday_date = yesterday.strftime("%Y-%m-%d")
    formatted_today_date = today.strftime("%Y-%m-%d")
    apple_data = yf.download("AAPL",start=today, auto_adjust=True, interval="2m")
    total_data, y_data = training.sorting_input_data(apple_data)
    predict(total_data,capital)
    print(f"Total ${capital.capital}")
    print(f"Stocks held: {capital.stocks_held}")
    
def get_current_stock_price():
    symbol = 'AAPL'

    url = f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={API_KEY}'
    response = requests.get(url)
    data = response.json()

    current_price = data['c']
    return current_price
    
def main():
    start_time = datetime.datetime.now()
    interval = datetime.timedelta(minutes=TIME_INTERVAL)
    capital = Currency(10000.41405,0,0.0)
    
    running(start_time,interval,capital)
    
    flag = True
    while flag:
        current_time = datetime.datetime.now()
        if current_time - start_time >= interval:
            start_time = current_time
            running(start_time,interval,capital)
        if current_time.hour > 15 and current_time.minute > 59:
            #sell off stocks for the day
            current_price = get_current_stock_price()
            for stock in range(capital.stocks_held):
                capital.capital += current_price - ((current_price - capital.price_bought_at)*0.1)
            print(f"sell at: ${current_price}")
            flag = False
            
if __name__ == "__main__":
    main()
import requests

API_KEY = 'cno8r09r01qu79g549mgcno8r09r01qu79g549n0'
symbol = 'AAPL'

url = f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={API_KEY}'
response = requests.get(url)
data = response.json()

current_price = data['c']
print(f'Current AAPL stock price: ${current_price}')
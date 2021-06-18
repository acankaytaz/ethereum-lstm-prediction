# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 14:11:43 2021

@author: acan
"""
# pip install python-binance yukle cmd base
# IMPORTS
import pandas as pd
import math
import os.path
import time
from binance.client import Client
from datetime import datetime
from dateutil import parser

### API
binance_api_key = '[ .. ]'    
binance_api_secret = '[ .. ]' 

### CONSTANTS
binsizes = {"1m": 1, "5m": 5, "1h":60, "2h": 120, "4h":240, "12h":720, "1d": 1440}
batch_size = 750
binance_client = Client(api_key=binance_api_key, api_secret=binance_api_secret)


### FUNCTIONS
def minutes_of_new_data(symbol, kline_size, data, source):
    if len(data) > 0:  old = parser.parse(data["timestamp"].iloc[-1])
    elif source == "binance": old = datetime.strptime('1 Jan 2017', '%d %b %Y')
    if source == "binance": new = pd.to_datetime(binance_client.get_klines(symbol=symbol, interval=kline_size)[-1][0], unit='ms')
    return old, new

def get_all_binance(symbol, kline_size, save = False):
    filename = '%s-%s-data.csv' % (symbol, kline_size)
    if os.path.isfile(filename): data_df = pd.read_csv(filename)
    else: data_df = pd.DataFrame()
    oldest_point, newest_point = minutes_of_new_data(symbol, kline_size, data_df, source = "binance")
    delta_min = (newest_point - oldest_point).total_seconds()/60
    available_data = math.ceil(delta_min/binsizes[kline_size])
    if oldest_point == datetime.strptime('1 Jan 2017', '%d %b %Y'): print('%s icin mevcut olan %s lik aralıklı bilgileri saglaniyor, lutfen bekleyin...' % (symbol, kline_size))
    else: print('%s icin %d dakikalik yeni bilgileri %s zaman aralıklı olarak %d yeni ornek eklenecek sekilde Binance tarafindan saglaniyor..' % (symbol, delta_min, kline_size, available_data))
    klines = binance_client.get_historical_klines(symbol, kline_size, oldest_point.strftime("%d %b %Y %H:%M:%S"), newest_point.strftime("%d %b %Y %H:%M:%S"))
    data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    if len(data_df) > 0:
        temp_df = pd.DataFrame(data)
        data_df = data_df.append(temp_df)
    else: data_df = data
    data_df.set_index('timestamp', inplace=True)
    if save: data_df.to_csv(filename)
    print('Bilgiler basariyla toplandı!')
    return data_df


binance_symbols = ["ETHUSDT"] #btc cekmek icin "BTCUSD" eklenebilir

#for symbol in binance_symbols: #birden fazla coini tek loop icinde cekmek
# varilableXyz =  get_all_binance(symbol, '2h', save = True)   
   

    
    

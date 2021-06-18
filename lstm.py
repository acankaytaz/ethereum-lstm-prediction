# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:12:49 2021

@author: acan
"""
from main import *
import matplotlib.pyplot as plt
import numpy as np
#model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler

#ethereum datalarını 2017den günümüze kadar cekmek
etherData = get_all_binance("ETHUSDT","1d",save = True)
etherData.tail(5)

#veriler = pd.read_csv('ETHUSDT-2h-data.csv')
veriler = pd.read_csv('ETHUSDT-1d-data.csv')

#spearman 
spearman1 = veriler.corr(method ='spearman') #sütunların birbirleri ile 2li olarak spearman korelasyon değerleri

#alakasız sutunları silme
veriler.pop('quote_av') 
veriler.pop('close_time')
veriler.pop('tb_base_av')
veriler.pop('tb_quote_av')
veriler.pop('ignore')
veriler.pop('volume')
veriler.pop('trades')
veriler.pop('open')
veriler.pop('high')
veriler.pop('low')
#---------------------------------------------------

#machine learning
#data preprocessing 

scaler = MinMaxScaler()
close_price = veriler.close.values.reshape(-1,1)
scaled_price = scaler.fit_transform(close_price)  #normalized between 0-1

#train
print("---------------------- \n")
def processData(veriler,lb):
    X,Y = [],[]
    for i in range(len(veriler)-lb-1):
        X.append(veriler[i:(i+lb),0])
        Y.append(veriler[(i+lb),0])
    return np.array(X),np.array(Y)
  
lb=100
X,y = processData(scaled_price,lb)
X_train,X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
y_train,y_test = y[:int(y.shape[0]*0.80)],y[int(y.shape[0]*0.80):]

#bolünmüs verilerin değerleri
# print(X_train.shape[0],X_train.shape[1])
# print(X_test.shape[0], X_test.shape[1])
# print(y_train.shape[0])
# print(y_test.shape[0])

#model
model = Sequential()
model.add(LSTM(256, input_shape=(lb,1))) 
# model.add(LSTM(32))
# model.add(Dropout(0.2))
 
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')

#Reshaping data for
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
 

history = model.fit(X_train,y_train,epochs=300,validation_data=(X_test,y_test), batch_size=64, verbose = 1) #shuffle=False?
model.summary() 

#train datasete göre predict sonuclari
plt.figure(figsize=(12,8))
Xt = model.predict(X_train)
plt.plot(scaler.inverse_transform(y_train.reshape(-1,1)), label="Actual")
plt.plot(scaler.inverse_transform(Xt), label="Predicted")
plt.ylabel("fiyat $",fontsize=12)
plt.xlabel("veri sayisi",fontsize=12)
plt.legend()
plt.title("Train Dataset")

#test datasete göre predict sonuclari
plt.figure(figsize=(12,8))
Xt = model.predict(X_test)
plt.plot(scaler.inverse_transform(y_test.reshape(-1,1)), label="Actual")
plt.plot(scaler.inverse_transform(Xt), label="Predicted")
plt.ylabel("fiyat $",fontsize=12)
plt.xlabel("veri sayisi",fontsize=12)
plt.legend()
plt.title("Test Dataset")


#gelecek tahmini bölümü 
# print('x test verisi buyuklugu:',len(X_test))
# print('y test verisi buyuklugu:',len(y_test))

x_input = y_test[-100:].reshape(1, -1) #test datasının son 100 verisi
x_input.shape
# print(x_input.shape) #(1,100)

temp_input = list(x_input)
temp_input = temp_input[0].tolist()
# print(temp_input)

lstm_output = []
adimlar=100;
i=0;
while(i<30): #30 gun
    if(len(temp_input) > 100):
        #print(temp_input) comment
        x_input=np.array(temp_input[1:]) #saga bir kaydırma, yeni tahmin verilerini sona ekleyip baştan 1 ilerliyor
        # print("{} day input {}".format(i,x_input)) #isleme giren 100 veri
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1,adimlar,1))
        #print(x_input) 
        yhat = model.predict(x_input, verbose=0) #tahmin
        print("+{}. gün ciktisi {}".format(i,yhat)) #100 verinin predict'e girdikten sonraki günün tahmin sonucu
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lstm_output.extend(yhat.tolist())
        i = i+1
    else:
        x_input = x_input.reshape(1, adimlar, 1)
        yhat = model.predict(x_input, verbose=0 )
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        lstm_output.extend(yhat.tolist())
        i = i+1
        
print("--------------------------\n")

print("30 gunluk $ değerinde tahmin verileri: \n",scaler.inverse_transform(np.array(lstm_output).reshape(-1,1)))

            
#prediction grafigi
plt.figure(figsize=(12,8))
plt.plot(scaler.inverse_transform(np.array(lstm_output).reshape(-1,1)), label="prediction", color='green')
plt.ylabel("fiyat $",fontsize=12)
plt.xlabel("+gün sayisi",fontsize=12)
plt.legend()
plt.title("30 gunluk tahmin")


















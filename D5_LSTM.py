# Skrypt do D+5 LSTM
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta, date
from pathlib import Path
import yfinance as yf
import openpyxl
import pickle
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import logging

logging.basicConfig(level=logging.INFO)

today = date.today()
comm_dict2 = {'^DJI':'DJI30'','EURUSD=X':'USD_EUR','CNY=X':'USD/CNY','BZ=F':'Brent_Oil','GC=F':'Gold','^IXIC':'NASDAQ',
             '^GSPC':'SP_500','^TNX':'10_YB','HG=F':'Copper','GBPUSD=X':'USD_GBP','JPY=X':'USD_JPY',
              'EURPLN=X':'EUR/PLN','PLN=X':'PLN/USD', 'AED=X':'USD/AED','^FVX':'5_YB','RUB=X':'USD/RUB',
              'PL=F':'Platinum','SI=F':'Silver','NG=F':'Natural Gas',
              'ZR=F':'Rice Futures','ZS=F':'Soy Futures','KE=F':'KC HRW Wheat Futures'}

def model_f(past):
    m_tab = None  # Inicjalizacja zmiennej m_tab
    for label, name in comm_dict2.items():
        col_name = {'Close': name}
        y1 = pd.DataFrame(yf.download(label, start='2003-12-01', end=today))[-past:]
        y1.reset_index(inplace=True)
        y11 = y1[['Date','Close']]
        y11.rename(columns=col_name, inplace=True)
        y2 = y1[['Close']]
        y2 = pd.DataFrame(y2.reset_index(drop=True))
        y2.rename(columns=col_name, inplace=True)
      
        if m_tab is None:
            m_tab = y11  # Tworzenie tabeli w pierwszym przebiegu
        else:
            m_tab = pd.concat([m_tab, y2], axis=1)

    m_tab.fillna(0)
    m_tab.to_pickle('Nm_data.pkl')
        
def data_set_eur():
    eur_df = pd.read_pickle('Nm_data.pkl')
    var_eur = eur_df['EUR/PLN']
    dict_eur = eur_df[['DJI30', 'USD_EUR', 'USD/CNY', 'Brent_Oil', 'Gold', 'NASDAQ', 'SP_500',
                   '10_YB', 'Copper', 'USD_GBP', 'USD_JPY', 'PLN/USD', '5_YB','USD/AED',
                   'USD/RUB', 'Platinum', 'Silver', 'Natural Gas', 'Rice Futures',
                   'Soy Futures', 'KC HRW Wheat Futures']]
    eur_rr = (dict_eur - dict_eur.shift(1)) / dict_eur.shift(1)  # ta linijka kodu liczy wszystkie stopy zwrotu
    eur_rr_df = pd.concat([var_eur, eur_rr], axis=1)
    eur_rr_f = eur_rr_df.dropna()

    n_rr_eur = eur_rr_f[['EUR/PLN', 'DJI30', 'USD_EUR', 'USD/CNY', 'Brent_Oil',
                        'Gold', 'NASDAQ', 'SP_500', '10_YB', 'Copper', 'USD_GBP', 'USD_JPY','USD/AED',
                        'PLN/USD', '5_YB', 'USD/RUB', 'Platinum', 'Silver', 'Natural Gas',
                        'Rice Futures', 'Soy Futures', 'KC HRW Wheat Futures']]
    n_rr_eur.fillna(0)
    n_rr_eur.to_pickle('n_rr_eur.pkl')  
    
def LSTM_D5_Model(data_set):
    set_1 = data_set.fillna(0)
    scaler=MinMaxScaler(feature_range=(0,1))
    set_1_scaled = scaler.fit_transform(np.array(set_1)) #set_1_scaled.min(axis=0) #set_1_scaled.max(axis=0)
    set_1_scaled.min(axis=0), set_1_scaled.max(axis=0)
    
    tr_size=int(len(set_1_scaled)*0.7)
    te_size=len(set_1_scaled)-tr_size
    tr_data = set_1_scaled[0:tr_size,:]
    te_data = set_1_scaled[tr_size : tr_size+te_size,:]  # why do we need , : ? [tr_size : tr_size+te_size,:] this cut the last part of array
    def create_dataset(dataset, time_step): # time_step=1
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-5):  # tutaj ustalamy na ile okresów w przód prognozujemy
            a = dataset[i:(i+time_step)]
            dataX.append(a)
            dataY.append(dataset[(i + time_step):(i + time_step+5), 0]) # tutaj ustalamy którą zmienną prognozujemy, oraz na ile okresów w przód

        return np.array(dataX), np.array(dataY)
   
    time_step = 200  # to jest krytyczny parametr
    X_train, y_train = create_dataset(tr_data, time_step)
    X_test, y_test = create_dataset(te_data, time_step)
    model = Sequential()

    model.add(LSTM(200,return_sequences=True,input_shape=(200,22))) # liczba kolumn musi tutaj być taka sama jak w X_train
    model.add(LSTM(200,return_sequences=True))
    model.add(LSTM(200))
    model.add(Dense(5))
    # model.add(Dense(5)) # tutaj musi być wprowadzona liczba dni na które robimy prognozę
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size = 64, verbose = 2) # 250 give better results
    train_data = model.evaluate(X_train, y_train, verbose=0) 
    test_data = model.evaluate(X_test, y_test, verbose=0)
    train_predi = model.predict(X_train, verbose=1)
    test_predi = model.predict(X_test, verbose=1)
    set150 = set_1[-200:].to_numpy() #set125 = np.array([set1[-125:].to_numpy()])
    set_150_sc = scaler.transform(set150) #set_125_sc.shape
    set_150_scaled = np.array([set_150_sc]) # set_125_scaled.shape
    forecast = model.predict(set_150_scaled, verbose=1)
    _forecast = (forecast - scaler.min_[0])/scaler.scale_[0]
    _forecast_df = pd.DataFrame(_forecast)
    _forecast_df.to_pickle('_fore_D5.pkl')
    
def D5_eur_forecast():
    D5_eur_F = pd.read_pickle('_fore_D5.pkl')
    D5_eur_fore = D5_eur_F.T
    D5_eur_fore.rename(columns={0: 'EUR_FOR'}, inplace=True)
    D5_eur_fore
    def get_next_workdays(start_date, num_days):
        workdays = []
        current_date = start_date
        while len(workdays) < num_days:
            if current_date.weekday() < 5:  # Poniedziałek=0, Niedziela=6
                workdays.append(current_date.date())
            current_date += timedelta(days=1)
        return workdays

    today_date = datetime.today()
    workdays = get_next_workdays(today_date, 5)

    if 'Date' in D5_eur_fore.columns:
        D5_eur_fore.drop(columns=['Date_EUR'], inplace=True)

    D5_eur_fore.insert(0, 'Date_EUR', workdays + [None] * (len(D5_eur_fore) - len(workdays)))
    D5_eur_fore.to_pickle('D5_eur_fore.pkl')
    
def D5_tabel():
    D5_eur_tabel = pd.read_pickle('D5_eur_tabel.pkl')
    D5_eur_fore = pd.read_pickle('D5_eur_fore.pkl')
    new_rows = pd.DataFrame({'Date': D5_eur_fore['Date_EUR'],'EUR/PLN': [None] * len(D5_eur_fore), 
                             'Day + 5 Prediction': D5_eur_fore['EUR_FOR'] })
    D5_eur_tabel = pd.concat([D5_eur_tabel, new_rows], ignore_index=True)
    eur_df = pd.read_pickle('Nm_data.pkl')
    data = eur_df['Date'].iloc[-1]
    k_w = eur_df['EUR/PLN'].iloc[-1]
    D5_eur_tabel.loc[D5_eur_tabel['Date'] == data, 'EUR/PLN'] = k_w
    D5_eur_tabel.to_pickle('D5_eur_tabel.pkl')
    
def run_D5_model():
    model_f(3001)
    data_set_eur()
    eur_df = pd.read_pickle('n_rr_eur.pkl')
    LSTM_D5_Model(eur_df)
    D5_eur_forecast()
    D5_tabel()

run_D5_model()
    

        
        

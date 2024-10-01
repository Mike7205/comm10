import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta, date
import pickle


def D5_tabel_daily ():
    D5_eur_tabel = pd.read_pickle('D5_eur_tabel.pkl')
    D5_eur_daily = pd.read_pickle('D1_EUR_a.pkl')
    data = D5_eur_daily['Date'].iloc[-1]
    k_w = D5_eur_daily['EUR/PLN'].iloc[-1]
    D5_eur_tabel.loc[D5_eur_tabel['Date'] == data, 'EUR/PLN'] = k_w
    D5_eur_tabel.to_pickle('D5_eur_tabel.pkl')
    
    
D5_tabel_daily ()
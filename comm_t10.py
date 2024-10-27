# To jest wersja 9 z Arima

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import pickle
from datetime import datetime, timedelta, date
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import appdirs as ad
import yfinance as yf
from sklearn.linear_model import LinearRegression
from streamlit import set_page_config
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Set page configuration for full width
st.set_page_config(layout="wide")

# start definicji strony
st.title('Selected global economy indicators')
st.subheader('& own LSTM prediction models', divider='blue')

# Definicje
today = date.today()
comm_dict = {'^GSPC':'SP_500','^DJI':'DJI30','^IXIC':'NASDAQ','000001.SS':'SSE Composite Index','^HSI':'HANG SENG INDEX','^VIX':'CBOE Volatility Index','^RUT':'Russell 2000',
             '^BVSP':'IBOVESPA','^FTSE':'FTSE 100','^GDAXI':'DAX PERFORMANCE-INDEX', '^N100':'Euronext 100 Index','^N225':'Nikkei 225',
             'EURUSD=X':'EUR_USD','EURCHF=X':'EUR_CHF','CNY=X':'USD/CNY', 'GBPUSD=X':'USD_GBP','JPY=X':'USD_JPY','EURPLN=X':'EUR/PLN','PLN=X':'PLN/USD','GBPPLN=X':'PLN/GBP', 
             'RUB=X':'USD/RUB','DX-Y.NYB':'US Dollar Index','^XDE':'Euro Currency Index', '^XDN':'Japanese Yen Currency Index',
             '^XDA':'Australian Dollar Currency Index','^XDB':'British Pound Currency Index','^FVX':'5_YB','^TNX':'10_YB','^TYX':'30_YB', 
             'CL=F':'Crude_Oil','BZ=F':'Brent_Oil', 'GC=F':'Gold','HG=F':'Copper', 'PL=F':'Platinum','SI=F':'Silver','NG=F':'Natural Gas',
              'ZR=F':'Rice Futures','ZS=F':'Soy Futures','BTC-USD':'Bitcoin USD','ETH-USD':'Ethereum USD'}

# Pobieranie danych
def comm_f(comm):
    global df_c1
    for label, name in comm_dict.items():
        if name == comm:
            df_c = pd.DataFrame(yf.download(f'{label}', start='2000-09-01', end = today,interval='1d'))
            df_c1 = df_c.reset_index()
           
    return df_c1   

# Dane historyczne                    
def comm_data(comm):
    global Tab_his
    shape_test=[]
    sh = df_c1.shape[0]
    start_date = df_c1.Date.min()
    end_date = df_c1.Date.max()
    close_max = "{:.2f}".format(df_c1['Close'].max())
    close_min = "{:.2f}".format(df_c1['Close'].min())
    last_close = "{:.2f}".format(df_c1['Close'].iloc[-1])
    v = (comm, sh, start_date,end_date,close_max,close_min,last_close)
    shape_test.append(v)
    Tab_length = pd.DataFrame(shape_test, columns= ['Name','Rows', 'Start_Date', 'End_Date','Close_max','Close_min','Last_close'])   
    Tab_his = Tab_length[['Start_Date','End_Date','Close_max','Close_min','Last_close']]
    Tab_his['Start_Date'] = Tab_his['Start_Date'].dt.strftime('%Y-%m-%d')
    Tab_his['End_Date'] = Tab_his['End_Date'].dt.strftime('%Y-%m-%d')
    #Tab_his1 = Tab_his.T
    #Tab_his1.rename(columns={0: "Details"}, inplace=True)
        
    return Tab_his

#definicja zakładki bocznej
st.html(
    """
<style>
[data-testid="stSidebarContent"] {color: black; background-color: #008AD8} 
</style>
""")
st.sidebar.subheader('Indexies, Currencies, Bonds, Commodities & Crypto', divider="grey")
comm = st.sidebar.radio('',list(comm_dict.values()))
comm_f(comm)
st.sidebar.write('© Michał Leśniewski')
st.sidebar.image('Footer3.gif', use_column_width=True)

# tu wstawimy wykresy 15 minutowe
def t1_f(char1):
    global tf_c1
    for label, name in comm_dict.items():
        if name == char1:
            box = yf.Ticker(label)
            tf_c = pd.DataFrame(box.history(period='1d', interval="1m"))
            tf_c1 = tf_c[-300:]
    return tf_c1 

def t2_f(char2):
    global tf_c2
    for label, name in comm_dict.items():
        if name == char2:        
            box = yf.Ticker(label)
            tf_c = pd.DataFrame(box.history(period='1d', interval="1m"))
            tf_c2 = tf_c[-300:]
    return tf_c2 

col1, col2 = st.columns([0.5, 0.5])
with col1:
    box = list(comm_dict.values())
    char1 = st.selectbox('Last 4 hours trading dynamics', box, index= box.index('Brent_Oil'),key = "<char1>")
    t1_f(char1)
    data_x1 = tf_c1.index
    fig_char1 = px.line(tf_c1, x=data_x1, y=['Open','High','Low','Close'],color_discrete_map={
                 'Open':'yellow','High':'red','Low':'blue','Close':'green'}, width=750, height=400) 
    fig_char1.update_layout(showlegend=False)
    fig_char1.update_layout(xaxis=None, yaxis=None)
    st.plotly_chart(fig_char1) #use_container_width=True
with col2:
    char2 = st.selectbox('Last 4 hours trading dynamics', box, index=box.index('PLN/USD'),key = "<char2>")
    t2_f(char2)
    data_x2 = tf_c2.index
    fig_char2 = px.line(tf_c2, x=data_x2, y=['Open','High','Low','Close'],color_discrete_map={
                 'Open':'yellow','High':'red','Low':'blue','Close':'green'}, width=750, height=400) 
    fig_char2.update_layout(showlegend=True)
    fig_char2.update_layout(xaxis=None, yaxis=None)
    st.plotly_chart(fig_char2)

# Definicja wykresu średnich ruchomych 
st.subheader(f'Short and long rolling averages signals for -> {comm}', divider='blue')
side_tab = pd.DataFrame(comm_data(comm))
st.write('Main Metrics:')
st.markdown(side_tab.to_html(escape=False, index=False), unsafe_allow_html=True)
xy = (list(df_c1.index)[-1] + 1)  
col3, col4, col5 = st.columns([0.4, 0.3, 0.3])
with col3:
    oil_p = st.slider('How long prices history you need?', 0, xy, 200, key = "<commodities>")
with col4:
    nums = st.number_input('Enter the number of days for short average',value=10, key = "<m30>")
with col5:
    numl = st.number_input('Enter the number of days for long average',value=30, key = "<m35>")
    
def roll_avr(nums,numl):
    global df_c_XDays
    # Oblicz krótkoterminową i długoterminową średnią kroczącą
    df_c1['Short_SMA']= df_c1['Close'].rolling(window=nums).mean()
    df_c1['Long_SMA']= df_c1['Close'].rolling(window=numl).mean()
    
    # Generuj sygnały kupna i sprzedaży
    df_c1['Buy_Signal'] = (df_c1['Short_SMA'] > df_c1['Long_SMA']).astype(int).diff()
    df_c1['Sell_Signal'] = (df_c1['Short_SMA'] < df_c1['Long_SMA']).astype(int).diff()
     
    df_c_XDays = df_c1.iloc[xy - oil_p:xy]
      
    fig1 = px.line(df_c_XDays, x='Date', y=['Close','Short_SMA','Long_SMA'], color_discrete_map={'Close':'#d62728',
                  'Short_SMA': '#F39F18','Long_SMA':'#0d0887'}, width=1000, height=500)
    fig1.add_trace(go.Scatter(x=df_c_XDays[df_c_XDays['Buy_Signal'] == 1].Date, y=df_c_XDays[df_c_XDays['Buy_Signal'] == 1]['Short_SMA'], name='Buy_Signal', mode='markers', 
                             marker=dict(color='green', size=15, symbol='triangle-up')))
    fig1.add_trace(go.Scatter(x=df_c_XDays[df_c_XDays['Sell_Signal'] == 1].Date, y=df_c_XDays[df_c_XDays['Sell_Signal'] == 1]['Short_SMA'], name='Sell_Signal',
                              mode='markers', marker=dict(color='red', size=15, symbol='triangle-down')))
    buy_signals = df_c_XDays[df_c_XDays['Buy_Signal'] == 1]
    #for i in buy_signals.index:
    #    fig1.add_hline(y=buy_signals.loc[i, 'Short_SMA'], line_width=0.5, line_dash="dash", line_color="black")

    sell_signals = df_c_XDays[df_c_XDays['Sell_Signal'] == 1]
    #for i in sell_signals.index:
    #    fig1.add_hline(y=sell_signals.loc[i, 'Short_SMA'], line_width=0.5, line_dash="dash", line_color="black")
    
    fig1.update_layout(xaxis=None, yaxis=None)
    st.plotly_chart(fig1, use_container_width=True)

roll_avr(nums,numl)

# Definicja wykresu stochastycznego 
st.subheader(f'Stochastic oscillator signals for -> {comm}', divider='blue')

xyx = (list(df_c1.index)[-1] + 1)  
col6, col7, col8 = st.columns([0.4, 0.3, 0.3])
with col6:
    cut_p = st.slider('How long prices history you need?', 0, xyx, 200, key = "<commodities1>")
with col7:
    K_num = st.number_input('Enter the number of days for %K parameter',value=14, key = "<k14>")
with col8:
    D_num = st.number_input('Enter the number of days for %D parameter',value=14, key = "<d14>")

# Obliczanie %K i %D dla oscylatora stochastycznego
def stoch_oscil(K_num,D_num):
    low_min  = df_c1['Low'].rolling(window = K_num).min()
    high_max = df_c1['High'].rolling(window = D_num).max()
    df_c1['%K'] = (100*(df_c1['Close'] - low_min) / (high_max - low_min)).fillna(0)
    df_c1['%D'] = df_c1['%K'].rolling(window = 3).mean()

    # Generowanie sygnałów kupna/sprzedaży
    df_c1['Buy_Signal'] = np.where((df_c1['%K'] < 20) & (df_c1['%K'] > df_c1['%D']), df_c1['Close'], np.nan)
    df_c1['Sell_Signal'] = np.where((df_c1['%K'] > 80) & (df_c1['%K'] < df_c1['%D']), df_c1['Close'], np.nan)

    df_cx_d = df_c1.iloc[xyx - cut_p:xyx]

    fig2 = px.line(df_cx_d,x='Date', y=['Close'],color_discrete_map={'Close':'dodgerblue'}, width=1000, height=500) #'Close':'#d62728',,'%K': '#f0f921','%D':'#0d0887'
    fig2.add_trace(go.Scatter(x=df_cx_d['Date'], y=df_cx_d['Buy_Signal'], mode='markers', name='Buy Signal', marker=dict(color='green', size=15, symbol='triangle-up')))
    fig2.add_trace(go.Scatter(x=df_cx_d['Date'], y=df_cx_d['Sell_Signal'], mode='markers', name='Sell Signal', marker=dict(color='red', size=15, symbol='triangle-down')))

    # Dodajemy poziome linie dla sygnałów kupna i sprzedaży
    buy_signals = df_cx_d.dropna(subset=['Buy_Signal'])
    #for i in buy_signals.index:
    #    fig2.add_hline(y=buy_signals.loc[i, 'Buy_Signal'], line_width=0.5, line_dash="dash", line_color="black")

    sell_signals = df_cx_d.dropna(subset=['Sell_Signal'])
    #for i in sell_signals.index:
    #    fig2.add_hline(y=sell_signals.loc[i, 'Sell_Signal'], line_width=0.5, line_dash="dash", line_color="black")

    fig2.update_layout(xaxis=None, yaxis=None)
    st.plotly_chart(fig2, use_container_width=True)

stoch_oscil(K_num,D_num)

# Arima - model - prognoza trendu
def Arima_f(comm, size_a):
    data = np.asarray(df_c1['Close'][-300:]).reshape(-1, 1)
    p = 10
    d = 0
    q = 5
    n = size_a

    model = ARIMA(data, order=(p, d, q))
    model_fit = model.fit(method_kwargs={'maxiter': 3000})
    model_fit = model.fit(method_kwargs={'xtol': 1e-6})
    fore_arima = model_fit.forecast(steps=n)  
    
    arima_dates = [datetime.today() + timedelta(days=i) for i in range(0, size_a)]
    arima_pred_df = pd.DataFrame({'Date': arima_dates, 'Predicted Close': fore_arima})
    arima_pred_df['Date'] = arima_pred_df['Date'].dt.strftime('%Y-%m-%d')
    arima_df = pd.DataFrame(df_c1[['Date','High','Close']][-500:])
    arima_df['Date'] = arima_df['Date'].dt.strftime('%Y-%m-%d')
    arima_chart_df = pd.concat([arima_df, arima_pred_df], ignore_index=True)
    x_ar = (list(arima_chart_df.index)[-1] + 1)
    arima_chart_dff = arima_chart_df.iloc[x_ar - 30:x_ar]
    
    fig_ar = px.line(arima_chart_dff, x='Date', y=['High', 'Close', 'Predicted Close'], color_discrete_map={
                  'High': 'yellow', 'Close': 'black', 'Predicted Close': 'red'}, width=1000, height=500)
    fig_ar.add_vline(x = today,line_width=3, line_dash="dash", line_color="green")
    fig_ar.update_layout(xaxis=None, yaxis=None)
    st.plotly_chart(fig_ar, use_container_width=True)      
    
# definicja wykresu obortów
def vol_chart(comm):
    volc = ['Crude_Oil','Brent_Oil','Gold','Copper','Platinum','Silver','Natural Gas','Rice Futures','Soy Futures','KC HRW Wheat Futures']
    if comm in volc:

        Co_V = df_c1[['Date', 'Volume']]
        Co_V['Co_V_M']= Co_V['Volume'].rolling(window=90).mean().fillna(0)
        V_end = (list(Co_V.index)[-1] + 1)

        st.subheader(comm+' Volume', divider='blue')
        Vol = st.slider('How long prices history you need?', 0, V_end, 200, key = "<volume>") 
        Co_V_XD = Co_V.iloc[V_end - Vol:V_end]

        fig3 = px.area(Co_V_XD, x='Date', y='Volume',color_discrete_map={'Volume':'#1f77b4'})
        fig3.add_traces(go.Scatter(x= Co_V_XD.Date, y= Co_V_XD.Co_V_M, mode = 'lines', line_color='red'))
        fig3.update_traces(name='90 Days Mean', showlegend = False)

        st.plotly_chart(fig3, use_container_width=True)
     
vol_chart(comm)

# Definicja modelu predykcyjnego Random Forest
st.subheader(f'Random Forest Model predictions for -> {comm}', divider='blue')
forest = st.slider('Prediction for how many days ?', 1, 1, 10, key = "<forest>") 

@st.cache_resource
def model_forest(past):    
    import warnings
    warnings.filterwarnings("ignore")
    m_tab = None  # Inicjalizacja zmiennej m_tab

    for label, name in comm_dict.items():
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
    m_tab.to_pickle('Forest_data.pkl')

def forest_cut_variables(comm):
    forest_cor = pd.read_pickle('Forest_data.pkl')
    forest_cut = forest_cor.drop(['Date'], axis=1)
    correlations = forest_cut.corr()
    comm_correlations = correlations[comm]
    filtered_columns = comm_correlations[(comm_correlations >= 0.05) & (comm_correlations <= 0.65)].index
    filtered_forest_cor = forest_cut[[comm] + list(filtered_columns)]
    fil_forest_cor = pd.concat([forest_cor['Date'],filtered_forest_cor], axis=1)
    fil_forest_cor = fil_forest_cor.fillna(0)
    fil_forest_cor.to_pickle('fil_forest_cor.pkl')

def data_set_forest(comm):
    forest_df = pd.read_pickle('fil_forest_cor.pkl')
    var_described = forest_df[comm]
    describing_vars = forest_df.drop(['Date',comm], axis=1)
    describing_rr = (describing_vars - describing_vars.shift(1)) / describing_vars.shift(1)  # ta linijka kodu liczy wszystkie stopy zwrotu
    forest_rr_df = pd.concat([forest_df['Date'],var_described, describing_rr], axis=1)
    forest_rr_f = forest_rr_df.fillna(0)
    forest_rr_f.to_pickle('forest_rr_f.pkl')

def rand_forest(comm, forest):
    new_ranf = pd.read_pickle('forest_rr_f.pkl')
    X = new_ranf.drop(columns=['Date',comm])
    y = new_ranf[comm]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    new_data = X[-200:]
    prediction = model.predict(new_data)
    #prediction[-forest:]

    from pandas.tseries.offsets import BDay
    new_ranp = new_ranf[['Date', comm]][-200:]
    forest_p = new_ranp.copy()
    forest_p['Predicted Close'] = prediction[:200]
    last_date = forest_p['Date'].iloc[-1]
    future_dates = [last_date + BDay(i) for i in range(1, forest+1)]
    future_data = pd.DataFrame({'Date': future_dates,comm: [np.nan] * len(future_dates),
        'Predicted Close': prediction[-forest:]})
    forest_p = pd.concat([forest_p, future_data], ignore_index=True)
    forest_p['Predicted Close Mean']= forest_p['Predicted Close'].rolling(window = forest).mean()
    fig_forest = px.line(forest_p, x='Date', y=[comm, 'Predicted Close Mean'],color_discrete_map={
                     comm:'green','Predicted Close Mean':'red'}, width=1000, height=500)             #'Predicted Close', ,'Predicted Close':'red'
    fig_forest.update_layout(showlegend=True,xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='Lightgrey'), #plot_bgcolor='white',
                          yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='Lightgrey'))
    st.plotly_chart(fig_forest, use_container_width=True)

model_forest(1001)
forest_cut_variables(comm)
data_set_forest(comm)
rand_forest(comm, forest)

# Definicja tabelki z artykułami z Yahoo
#try:
#    st.subheader(f'Yahoo news related to {comm}', divider='green')
#    t_comm = next(key for key, value in comm_dict.items() if value == comm)
#    data = yf.Ticker(t_comm)
#    news = data.news
    
#    if news:
#        news_data = [{"Title": item['title'], 
#                      "Link": f"<a href='{item['link']}' target='_blank'>{item['link']}</a>",
#                      "Publisher": item['publisher']} for item in news]

#        table = "<table><tr><th>Title</th><th>Link</th><th>Publisher</th></tr>"
#        for item in news_data:
#            row = f"<tr><td>{item['Title']}</td><td>{item['Link']}</td><td>{item['Publisher']}</td></tr>"
#            table += row
#        table += "</table>"

#        st.markdown(table, unsafe_allow_html=True)
    
#    else:
#        st.info("There is no relevant news for this topic")  
    
#except KeyError:
#    st.error("KeyError: Symbol not found in Yahoo Finance.")
#except Exception as e:
#    st.error(f"An unexpected error occurred: {e}")
    
st.subheader('Prediction models and benchmarks', divider='blue')
def D5_tabel_date(): # Konwersja kolumny 'Date' na typ daty
    D1EUR_df = pd.read_pickle('D1_EUR_a.pkl')
    D5T_df = pd.read_pickle('D5_eur_tabel.pkl')

    D5T_df['Date'] = pd.to_datetime(D5T_df['Date'])
    D1EUR_df['Date'] = pd.to_datetime(D1EUR_df['Date'])

    for index, row in D5T_df.iterrows():
        if pd.isna(row['EUR/PLN']):
            date_to_find = row['Date']
            matching_row = D1EUR_df.loc[D1EUR_df['Date'] == date_to_find]
            if not matching_row.empty:
                value_to_copy = matching_row['EUR/PLN'].values[0]
                D5T_df.at[index, 'EUR/PLN'] = value_to_copy
                
    D5T_df.to_pickle('D5_eur_tabel.pkl')

col9, col10, col11, col12 = st.columns(4)
with col9:
    checkbox_value3 = st.checkbox('Arima model trend prediction for x days',key = "<arima_m>")

if checkbox_value3:
    st.subheader(f'{comm} Arima model prediction', divider='grey')
    size_a = st.radio('Prediction for ... days ?: ', [5,4,3,2,1], horizontal=True, key = "<arima21>")
    Arima_f(comm,size_a)    

with col10:
    checkbox_value2 = st.checkbox('Own LSTM EUR/PLN D+5 prediction model',key = "<lstm1>")

if checkbox_value2:
    D5_tabel_date()
    st.subheader('EUR/PLN exchange rate (D+5) predictions')
    val_D5E = pd.read_pickle('D5_eur_tabel.pkl')
    val_D5EP = val_D5E[['Date','Day + 5 Prediction']][-100:]
    val_D5EU = pd.read_pickle('D5_eur_tabel.pkl')
    val_D5EUR = val_D5EU[['Date','EUR/PLN']][-100:]
    day_es = val_D5EUR.shape[0]

    st.subheader(f'Predictions for the last {day_es} days', divider='grey')
  
    fig_D5E = px.line(val_D5EP, x='Date', y=['Day + 5 Prediction'],color_discrete_map={'Day + 5 Prediction':'red'}, width=1500, height=500)
    fig_D5E.add_trace(go.Scatter(x=val_D5EUR['Date'], y=val_D5EUR['EUR/PLN'], mode='lines', name='EUR/PLN', line=dict(color='blue')))

    fig_D5E.update_layout(plot_bgcolor='white',showlegend=True,xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='Lightgrey'),
                      yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='Lightgrey'))
    #fig_D5E.add_vline(x = today,line_width=1, line_dash="dash", line_color="black")
    st.plotly_chart(fig_D5E)

with col11:
    checkbox_value4 = st.checkbox('Own EUR/PLN LSTM prediction model (D+1)',key = "<lstm2>")    
    
if checkbox_value4:
    st.subheader('EUR/PLN exchange rate (D+1) predictions')
    val = pd.read_pickle('D1_EUR_a.pkl')
    val_1 = val[['Date','EUR/PLN','Day + 1 Prediction']][-100:]      #.iloc[:-1]
    day_s = val_1.shape[0]

    st.subheader(f'Predictions for the last {day_s} days', divider='grey')

    fig_val = px.line(val_1, x='Date', y=['EUR/PLN','Day + 1 Prediction'],color_discrete_map={
                 'EUR/PLN':'blue','Day + 1 Prediction':'red'}, width=1000, height=500 ) 

    fig_val.update_layout(plot_bgcolor='white',showlegend=True,xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='Lightgrey'),
                      yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='Lightgrey'))
   
    st.plotly_chart(fig_val, use_container_width=True)
    
with col12:
    checkbox_value5 = st.checkbox('Own USD/PLN LSTM prediction model (D+1)',key = "<lstm3>")    
    
if checkbox_value5:
    st.subheader('USD/PLN exchange rate (D+1) predictions')
    val_s = pd.read_pickle('D1_USD_a.pkl')
    val_s1 = val_s[['Date','USD/PLN','Day + 1 Prediction']][-100:]      #.iloc[:-1]
    day_s1 = val_s1.shape[0]

    st.subheader(f'Predictions for the last {day_s1} days', divider='grey')

    fig_vals = px.line(val_s1, x='Date', y=['USD/PLN','Day + 1 Prediction'],color_discrete_map={
                 'USD/PLN':'#03B303','Day + 1 Prediction':'#D9017A'}, width=1000, height=500 ) # #89CFF0
    fig_vals.update_layout(plot_bgcolor='white',showlegend=True,xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='Lightgrey'),
                      yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='Lightgrey'))
   
    st.plotly_chart(fig_vals, use_container_width=True)

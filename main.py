# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pycaret.regression import *
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import xgboost as xgb

START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('')

stocks = ('BTC-USD', 'ETH-USD','ADA-USD')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = 1


@st.cache_data 
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data2=load_data(selected_stock)
print(data2)
future_days=1
data['Future_Price']=data[['Close']].shift(-future_days)
data=data[['Close','Future_Price']]
d=data.copy()
X=np.array(d[d.columns])
X=X[:len(data)-future_days]
y=np.array(d['Future_Price'])
y=y[:-future_days]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=0,shuffle=False)
train_data=pd.DataFrame(x_train,columns=d.columns)

test_data=pd.DataFrame(x_test,columns=d.columns)

regression_setup=setup(data=train_data,target='Future_Price',session_id=123,use_gpu=True)
best_model=compare_models(sort='r2')
model=create_model(best_model)
evaluate_model(model)
unseen_predictions=predict_model(model,data=test_data)
period = n_years * unseen_predictions.shape[0]
print(unseen_predictions)
st.subheader('Raw data')
st.write(data2.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data2['Date'], y=data2['Open'], name="open"))
	fig.add_trace(go.Scatter(x=data2['Date'], y=data2['Close'], name="close"))
	fig.layout.update(title_text='Time Series data', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df = data2[['Date','Close']]
df = df.rename(columns={"Date": "ds", "Close": "y"})
df.head()

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=period)



forecast = m.predict(future)
print(forecast)
st.subheader(f'Forecast {selected_stock} plot for {period} days by RNN')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)




forecast1 = m.predict(future)
for i in range(0,unseen_predictions.shape[0]-1):
	forecast1.at[data2.shape[0]+i+1,'yhat']=unseen_predictions.at[unseen_predictions.shape[0]-1-i,'prediction_label']

print(forecast1) 
st.subheader(f'Forecast {selected_stock} plot for {period} days by LSTM')
fig2 = plot_plotly(m, forecast1)
st.plotly_chart(fig2)



forecast2 = m.predict(future)
for i in range(0,unseen_predictions.shape[0]-1):
	forecast2.at[data2.shape[0]+i+1,'yhat']=unseen_predictions.at[i,'Future_Price']

print(forecast2) 
st.subheader(f'Forecast {selected_stock} plot for {period} days by XGBOOST')
fig3 = plot_plotly(m, forecast2)
st.plotly_chart(fig3)

def create_features(df2):
    df2 = df2.copy()
    df2['hour'] = df2.index.hour
    df2['dayofweek'] = df2.index.dayofweek
    df2['quarter'] = df2.index.quarter
    df2['month'] = df2.index.month
    df2['year'] = df2.index.year
    df2['dayofyear'] = df2.index.dayofyear
    return df2

df2 = data2[['Date','Close']]
df2 = df2.set_index('Date')
df2.index = pd.to_datetime(df2.index)
df2 = create_features(df2)

def add_lags(df2):
    target_map = df2['Close'].to_dict()
    df2['lag1'] = (df2.index - pd.Timedelta('364 days')).map(target_map)
    df2['lag2'] = (df2.index - pd.Timedelta('728 days')).map(target_map)
    df2['lag3'] = (df2.index - pd.Timedelta('1092 days')).map(target_map)
    return df2

df2 = add_lags(df2)

df2 = create_features(df2)

FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year', 'lag1','lag2','lag3']
TARGET = 'Close'

X_all = df2[FEATURES]
y_all = df2[TARGET]

reg = xgb.XGBRegressor(base_score=0.5,
                       booster='gbtree',    
                       n_estimators=500,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)
reg.fit(X_all, y_all,
        eval_set=[(X_all, y_all)],
        verbose=100)

# Create future dataframe
future = pd.date_range(TODAY,'2024-06-30', freq='5h')
future_df2 = pd.DataFrame(index=future)
future_df2['isFuture'] = True
df2['isFuture'] = False
df2_and_future = pd.concat([df2, future_df2])
df2_and_future = create_features(df2_and_future)
df2_and_future = add_lags(df2_and_future)

future_w_features = df2_and_future.query('isFuture').copy()
future_w_features['pred'] = reg.predict(future_w_features[FEATURES])
future_w_features.reset_index()




















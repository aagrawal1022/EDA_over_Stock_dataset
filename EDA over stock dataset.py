#!/usr/bin/env python
# coding: utf-8

# ## Brief description of the data set and a summary of its attributes
# 
# In this project, I have performed exploratory data analysis over NESTLE INDIA stock dataset traded over NSE in last 11 years.
# 
# link of stock - https://in.finance.yahoo.com/quote/NESTLEIND.NS?p=NESTLEIND.NS&.tsrc=fin-srch
# 
# Link to dataset - https://www.kaggle.com/rohanrao/nifty50-stock-market-data?select=NESTLEIND.csv
# 
# It contains 15 columns out of 7 will be mainly used for analysis which are described below
# (Each row defines a day of trading):
# 
# **Date** - Date on which trading occured
# 
# **Prev Close** - Closing price of stock traded on previous trading day
# 
# **Close** - Closing price of stock traded today
# 
# **Volume** - Volume is the number of shares of a security traded during a given period of time (in a day).
# 
# **VWAP** - Volume weighted average price. The volume weighted average price helps in comparing the current price of the stock to            a benchmark, making it easier for investors to make decisions on when to enter and exit the market.
# 
# **Deliverable Volume** - Deliverable Volume is the quantity of shares which actually move from one set of people (who had those shares in their demat account before today and are selling today) to another set of people (who have purchased those shares and will get those shares by T+2 days in their demat account).
# 
# **%Delivery** - It is defined as total Deliverable Volume divided by total Volume on particular day

# ## Initial plan for data exploration
# 
# As data contains many unneccessary columns which are not needed for our model we will drop them and using boxplot determing/handling outliers.

# In[68]:


#Importing necessay library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('pylab', 'inline')

get_ipython().run_line_magic('config', "InlineBackend.figure_formats = ['svg']")


# # RETRIEVING DATA

# In[69]:


df=pd.read_csv("NESTLEIND.csv")
df['Date']=pd.to_datetime(df['Date']) #Changing DATE column from object datatype to pandas datetime datatype.
df.set_index("Date",inplace=True) #Initially 0 to n is set as index but for time series analysis we need index to be date.


# In[70]:


df.head()


# In[71]:


#Plotting Graph for stock price movement over period of time
plt.figure(figsize=(5,4))
plt.plot(df["Close"])
plt.grid()
plt.title('Nestle India',fontsize=14,fontstyle='oblique')
plt.xlabel('Date',fontsize=8)
plt.ylabel('Close Price in INR',fontsize=8)


# In[72]:


#Plotting Candelstick Chart for better visulazation
import plotly.graph_objects as go
figure=go.Figure(
data=[
    go.Candlestick(
    x=df.index,
    low=df['Low'],
    high=df['High'],
    open=df['Open'],
        close=df['Close']
    )
])
figure.update_layout(height=900,title='Nestle India Stock Movement',width=950)
figure.show()


# ## Actions taken for data cleaning and feature engineering
# 1. Droping unnecessay columns
# 2. Checking for null values
# 3. determing and handling outliers
# 4. boxplot visulaztion after handling outliers
# 5. MinMaxScaling for feature scaling (In this code 'Close' and 'Prev Close' not scaled as they are used in hypothesis testing)
# 

# # DATA CLEANING

# In[73]:


#Removing Unnecessay Columns
data=df.drop(['Symbol','Series','Open','High','Low','Last','Trades','Turnover'],axis=1)
data


# In[74]:


#CHECKING FOR MISSING VALUE
print(data.isnull().any())


# In[75]:


print(data.isna().sum())


# #As there is no missing value in our data, we dont need to perform any replace,fillna,drop row operations

# In[76]:


#Checking Similrity b/w of VWAP with Close and Volume
from scipy.spatial import distance,distance_matrix
dst1=distance.minkowski(data['VWAP'].tolist(),data['Close'].tolist(),2)
dst2=distance.minkowski(data['VWAP'].tolist(),data['Volume'].tolist(),2)
dst3=distance.minkowski(data['Close'].tolist(),data['Volume'].tolist(),2)
print(dst1,dst2,dst3)


# This shows the VWAP vs Volume have almost same similarity as Stock price vs Volume have.

# OUTLIER DETECTION

# In[77]:


#1. VISULIZATION _ boxplot
fig,ax=plt.subplots(2,2,figsize=(12,8))
sns.boxplot(data['Volume'],ax=ax[0,0])
sns.boxplot(data['Deliverable Volume'],ax=ax[0,1])
sns.boxplot(data['Close'],ax=ax[1,0])
sns.boxplot(data['VWAP'],ax=ax[1,1])


#  As we can observe there is outlier present in volume and deliverable volumne we will use z score to remove it

# In[78]:


from scipy import stats
import numpy as np
vol = np.abs(stats.zscore(data))
print(np.where(vol>6))


# As we can there are top 20 values with extreme values which may cause problem in future therefore we can either remove the row or replace the extreme values with average
# value of the column.
# "

# In[79]:


#removing Data i.e OUTLIERS
data_new=data[(vol<6).all(axis=1)]


# In[80]:


data_new


# In[81]:


#2.RE- VISULIZATION _ boxplot
fig,ax=plt.subplots(2,2,figsize=(12,8))
sns.boxplot(data_new['Volume'],ax=ax[0,0])
sns.boxplot(data_new['Deliverable Volume'],ax=ax[0,1])
sns.boxplot(data_new['Close'],ax=ax[1,0])
sns.boxplot(data_new['VWAP'],ax=ax[1,1])


# # EDA

# In[82]:


data_new.describe()


# We can also remove Deliverable Voulme but
# 'The higher the Percent of Deliverable Quantity to Traded Quantity the better - it indicates that most buyers are expecting the price of the share to go up.'
# Therefore it can be useful feature.

# In[83]:


#TRANSORMING DATA
#Transformation cann't be done over categorical varibale, but as we don't have one we can apply over all required columns
cols=['Volume','VWAP','Deliverable Volume']
df_scaled = data_new.copy()
features = df_scaled[cols]


# In[84]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = data_new.copy()
df_scaled[cols] = scaler.fit_transform(features.values)
df_scaled.head()


# ### Key Findings and Insights, which synthesizes the results of Exploratory Data Analysis in an insightful and actionable manner
# 
# 1. 'Volume' and 'Deliverable Volume' contains outliers which were handled by removing that row
# 2.  VWAP vs Volume have almost same similarity as Stock price vs Volume have.
# 3. Average traded price is Rs. 7138.540461/-
# 4. data doesn't contain any NULL values

# In[85]:


df_scaled.reset_index(inplace=True)
df=df_scaled[df_scaled['%Deliverble']>0.9]
df


# In[86]:


a=df.index.tolist()
a=[x + 1 for x in a]
ini=df_scaled[df_scaled.index==a[0]]
for i in range(1,len(a)):
    ini=ini.append(df_scaled[df_scaled.index==a[i]])
ini


# In[87]:


count=0
for ind in range(0,len(ini.index)):
    if ini.iloc[ind]['Close']>=ini.iloc[ind]['Prev Close']:
        count+=1
print(count)


# ###  Formulating at least 3 hypothesis about this data
# 
# 1. Greater **Volume** represent greater intraday move of stock . (i.e. High-Low)
# 2. Stock has never fallen more than 20% from all time high and hit new all time high as long as.
# 3. whenever %delivery is higher than 50% stock has closed above previous day close.

# # Null Hypothesis
# 
# I claim that when % Delivery is greater than 0.8 next day price is closed above previous day has probablity >0.5.
# 
# How can we test it?
# 
# i have preformed an experiment and found that when %delivery is > .9 out of 212, 123 times i have been right i.e prob=0.58
# 
# p-value=5%

# 
# ### Determining the Null and Alternative Hypothesis
# 
# Null: **0.5 prob**       ---(It is asummed to be 0.5 as next day stock can close higher or lower than previous day is equally likely given other conditions remain constant)
# 
# Alternative: I am right, probability is **greater than 0.5**
# 

# In[88]:


cols=['Volume','VWAP','Deliverable Volume']
df_scaled = data_new.copy()
features = df_scaled[cols]
scaler = MinMaxScaler()
df_scaled = data_new.copy()
df_scaled[cols] = scaler.fit_transform(features.values)
df_scaled.reset_index(inplace=True)
df=df_scaled[df_scaled['%Deliverble']>0.8]
a=df.index.tolist()
a=[x + 1 for x in a]
ini=df_scaled[df_scaled.index==a[0]]
for i in range(1,len(a)):
    ini=ini.append(df_scaled[df_scaled.index==a[i]])
count=0
for ind in range(0,len(ini.index)):
    if ini.iloc[ind]['Close']>=ini.iloc[ind]['Prev Close']:
        count+=1
print(count,len(ini))


# In[89]:


from scipy.stats import binom
#318 is number of days where next day close is above previous day close
#604 is no. of days with delivery >.8
#0.5 is probability of closing above previous day
prob = 1 - binom.cdf(318, 604, 0.5)
print(str(round(prob*100, 1))+"%")


# The probability of getting close above previous day when %delivery is >0.8 is about 9.7%. This is more than 5%, so we can't reject the null and conclude that yes **Stock price will close above previous day close when %Delivery is greater than 0.8**.

# ###  Suggestions for next steps in analyzing this data
# 1. scale 'Prev Close' and 'Close' columns using MinMaxScaler
# 2. Use LSTM model for prediction as this is time series data
# 3. Visualzie using line chart 

# ### A paragraph that summarizes the quality of this data set and a request for additional data if needed
# 
# Traditionally and in order to predict market
# movement, investors used to analyze the
# stock prices and stock indicators in addition
# to the news related to these stocks.In this work, we
# propose an automated trading system that
# integrates mathematical functions, machine
# learning for the purpose of
# achieving better stock prediction accuracy
# and issuing profitable trades.
# 
# There is lakhs of stock avilable over which analysis can be performed as Fintech is hot subject and futher development will be appreciated.

# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### so that u dont have warnings
from warnings import filterwarnings
filterwarnings('ignore')


# ##### The tech stocks we'll use for this analysis

# In[3]:


path=r'C:\Users\Pranav\Downloads\sales data\Sales_Data-20230108T133928Z-001\individual_stocks_5yr-20230118T141908Z-001'
company_list = ['AAPL_data.csv', 'GOOG_data.csv', 'MSFT_data.csv', 'AMZN_data.csv']

#blank dataframe
all_data = pd.DataFrame()

for file in company_list:
    current_df = pd.read_csv(path+"/"+file)
    all_data = pd.concat([all_data, current_df])
    
all_data.shape


# In[4]:


all_data.head()


# In[5]:


all_data.dtypes


# In[6]:


all_data['date']=pd.to_datetime(all_data['date'])


# In[7]:


all_data['date'][0]


# In[8]:


all_data.columns


# ##### Analyse closing price of all the stocks

# In[11]:


tech_list = all_data['Name'].unique()


# In[12]:


plt.figure(figsize=(20,12))
for i, company in enumerate(tech_list,1):
    plt.subplot(2, 2, i)
    df=all_data[all_data['Name']==company]
    plt.plot(df['date'],df['close'])
    plt.title(company)

    


# In[ ]:





# ##### let's analyse the total volume of stock being traded each day

# In[13]:


plt.figure(figsize=(20,12))
for i, company in enumerate(tech_list,1):
    plt.subplot(2, 2, i)
    df=all_data[all_data['Name']==company]
    plt.plot(df['date'],df['volume'])
    plt.title(company)


# ##### using plotly

# In[14]:


import plotly.express as px


# In[1]:



for company in (tech_list):
    df=all_data[all_data['Name']==company]
    fig = px.line(df, x="date", y="volume", title=company)
    fig.show()


# In[ ]:





# In[16]:


all_data['Name'].unique()


# In[ ]:





# ##### analyse Daily price change in stock

# ##### Daily Stock Return Formula
#     To calculate how much you gained or lost per day for a stock, subtract the opening price from the closing price. Then, multiply the result by the number of shares you own in the company. 

# In[ ]:





# In[17]:


df=pd.read_csv('F:\EDA_projects\Stock_Data\individual_stocks_5yr\individual_stocks_5yr/AAPL_data.csv')
df.head()


# ##### percentage return

# In[18]:


df['1day % return']=((df['close']-df['open'])/df['close'])*100
df.head()


# In[19]:


df.columns


# ##### using plotly to visualise data

# In[18]:


import plotly.express as px
fig = px.line(df, x="date", y="1day % return", title='')
fig.show()


# ##### using matplotlib for visualisation

# In[19]:


plt.figure(figsize=(10,6))
df['1day % return'].plot()


# #### lets say between some interval

# In[20]:


df.set_index('date')['2016-01-01':'2016-03-31']['1day % return'].plot()
plt.xticks(rotation='vertical')


# #### Analyse monthly mean of close column

# In[21]:


df2=df.copy()


# In[22]:


df2['date']=pd.to_datetime(df2['date'])


# In[23]:


df2.set_index('date',inplace=True)


# In[24]:


df2.head()


# In[25]:


df2['close'].resample('M').mean().plot()


# ##### resampling close column year wise

# In[26]:


df2['close'].resample('Y').mean().plot()


# In[ ]:





# ##### Checking if the Stock prices of these tech companies(Amazon,Apple,Google,Microsoft) are correlated

# In[28]:


df2.head()


# ##### reading data of tech companies

# In[29]:


aapl=pd.read_csv('F:\EDA_projects\Stock_Data\individual_stocks_5yr\individual_stocks_5yr/AAPL_data.csv')
aapl.head()


# In[30]:


goog=pd.read_csv('F:\EDA_projects\Stock_Data\individual_stocks_5yr\individual_stocks_5yr/GOOG_data.csv')
goog.head()


# In[31]:


amzn=pd.read_csv('F:\EDA_projects\Stock_Data\individual_stocks_5yr\individual_stocks_5yr/AMZN_data.csv')
amzn.head()


# In[32]:


msft=pd.read_csv('F:\EDA_projects\Stock_Data\individual_stocks_5yr\individual_stocks_5yr/MSFT_data.csv')
msft.head()


# In[ ]:





# In[33]:


### create a blank dataframe
close=pd.DataFrame()


# In[34]:


close['aapl']=aapl['close']
close['goog']=goog['close']
close['amzn']=amzn['close']
close['msft']=msft['close']


# In[35]:


close.head()


# #### Multi-variate Analysis

# In[36]:


sns.pairplot(data=close)


# ##### co-relation plot for stock prices 

# In[37]:


sns.heatmap(close.corr(),annot=True)


# ###### Closing price of Google and Microsoft are well correlated
#        and Closing price of Amazon and Microsoft have a co-relation of 0.96

# In[ ]:





# ##### Analyse Daily return of each stock & how they are co-related

# In[38]:


data=pd.DataFrame()


# In[39]:


aapl.head()


# In[40]:


data['appl_change']=((aapl['close']-aapl['open'])/aapl['close'])*100
data['goog_change']=((goog['close']-goog['open'])/goog['close'])*100
data['amzn_change']=((amzn['close']-amzn['open'])/amzn['close'])*100
data['msft_change']=((msft['close']-msft['open'])/msft['close'])*100


# In[41]:


data.head()


# In[42]:


sns.pairplot(data=data)


# ##### Correlation plot for daily returns

# In[43]:


sns.heatmap(data.corr(),annot=True)


# ##### We can see that Amazon and microsoft have good correlation on daily returns

# In[44]:


data.head()


# In[45]:


data.columns


# In[46]:


type(data)


# #### Value at Risk analysis for Apple

# In[47]:


import seaborn as sns


# In[48]:


sns.distplot(data['appl_change'])


# ##### it somehow follows a normal distribution

# In[49]:


data['appl_change'].std()


# In[50]:


data['appl_change'].quantile(0.1)


# ##### 1.4246644227944307 means that 90% of the times the worst daily Loss will not exceed 1.42

# In[ ]:





# In[51]:


data.describe().T


# In[ ]:





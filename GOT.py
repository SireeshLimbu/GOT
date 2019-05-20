#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import itertools
plt.style.use('fivethirtyeight')
plt.rcParams.update({'figure.figsize':(15,7), 'figure.dpi':120})


# In[3]:


ov = pd.read_csv('overall.csv',names=['value'], header=0)      #Total Deaths per episode
df = pd.read_csv('named.csv',names=['value'], header=0)        #Named deaths per episode

import statistics
print('Mean number of deaths per episode')
print(statistics.mean(ov.value))


# In[8]:


############################################# Named Deaths #######################################################

# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df.value,linewidth=2.0); axes[0, 0].set_title('Original Series')
plot_acf(df.value, ax=axes[0, 1],linewidth=3.0)

# 1st Differencing
axes[1, 0].plot(df.value.diff(),linewidth=2.0); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df.value.diff().dropna(), ax=axes[1, 1],linewidth=3.0)

# 2nd Differencing
axes[2, 0].plot(df.value.diff().diff(),linewidth=2.0); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df.value.diff().diff().dropna(), ax=axes[2, 1],linewidth=3.0)

plt.show()


# In[9]:


#Dickey-Fuller Test for Stationarity
from statsmodels.tsa.stattools import adfuller
adf_test = adfuller(df.value)
 
print ("ADF = " + str(adf_test[0]))
print ("p-value = " +str(adf_test[1]))


# In[30]:


from statsmodels.tsa.arima_model import ARMA
# fit model
model2 = ARMA(df, order=(14, 0))
model_fit2 = model2.fit(disp=False)
# make prediction
yhat2 = model_fit2.predict(len(ov), len(ov))
print(yhat2)


# In[31]:


# plot residual errors
from pandas import DataFrame
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())


# In[76]:


##############################################Overall_Deaths#####################################################

# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(ov.value,linewidth=2.0); axes[0, 0].set_title('Original Series')
plot_acf(ov.value, ax=axes[0, 1],linewidth=3.0)

# 1st Differencing
axes[1, 0].plot(ov.value.diff(),linewidth=2.0); axes[1, 0].set_title('1st Order Differencing')
plot_acf(ov.value.diff().dropna(), ax=axes[1, 1],linewidth=3.0)

# 2nd Differencing
axes[2, 0].plot(ov.value.diff().diff(),linewidth=2.0); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(ov.value.diff().diff().dropna(), ax=axes[2, 1],linewidth=3.0)


plt.show()


# In[5]:


#Dickey-Fuller Test for Stationarity
from statsmodels.tsa.stattools import adfuller
 
adf_test = adfuller(ov.value)
 
print ("ADF = " + str(adf_test[0]))
print ("p-value = " +str(adf_test[1]))


# In[91]:


# fit model
model2 = ARMA(ov, order=(3, 0))
model_fit2 = model2.fit(disp=False)
# make prediction
yhat2 = model_fit2.predict(len(ov), len(ov))
print(yhat2)


# In[92]:


# plot residual errors
from pandas import DataFrame
residuals = DataFrame(model_fit2.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Stock Price prediction using the Long Short Term Memory neural networks
## We're using RNN to predict the closing prices of Stocks using the past 60 days


# In[5]:


import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[16]:


## Get the stock quotations

df = web.DataReader('TSLA', data_source = 'yahoo', start = '2012-01-01', end = '2021-01-12')
# show the data
df # This is the dataframe with the data on Tesla's stock price form Jan 2012 to Jan 2021


# In[10]:


# The shape of the data
df.shape


# In[17]:


# Visualize the stock price

plt.figure(figsize =(16,8))
plt.title("The history of Tesla")
plt.plot(df['Close'])
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close price in USD', fontsize = 18)
plt.show()


# In[19]:


# Create a dataframe with  only the "Close" column
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset)*.8)

training_data_len


# In[39]:


# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)


# In[42]:


# Create the scaled training dataset
# Craete the training dataset

train_data = scaled_data[0:training_data_len, :]
# Split the dataset into x_train and y_train

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    
    if i<=60:
        print(x_train)
        print(y_train)
        print()


# In[45]:


x_train, y_train = np.array(x_train), np.array(y_train)


# In[47]:


# Reshaping the data into the shape that's accepted by the LSTM


x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))


# In[77]:


# Building the LSTM neural network model

model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(units = 50, return_sequences = False))
model.add(Dense(units = 25))
model.add(Dense(units =1))          


# In[78]:


# Compiling the model

model.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[79]:


# Train the model
model.fit(x_train, y_train, batch_size=1, epochs = 2)


# In[80]:


# Test Data Set
test_data = scaled_data[training_data_len -60: , : ]

# Create the x_test and y_test datasets
x_test = []
y_test = dataset[training_data_len :, :] # Gets all of the rows from index 1818 to the rest and all of the columns

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])


# In[81]:


# Converting the x_test to a numpy array
x_test = np.array(x_test)


# In[82]:


# Reshape the data into the shape accepted by the LSTM
x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))


# In[83]:


#Getting the models predicted price values
predictions = model.predict(x_test) 


# In[84]:


# Undo Scaling
predictions = scaler.inverse_transform(predictions)


# In[85]:


#Calculate/Get the value of RMSE
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse


# In[86]:


#Plot/Create the data for the graph
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()


# In[87]:


#Show the valid and predicted prices
valid


# In[92]:


#Get the quote
tesla_quote = web.DataReader('TSLA', data_source='yahoo', start='2012-01-01', end = '2021-01-12')
#Create a new dataframe
new_df1 = tesla_quote.filter(['Close'])
#Get teh last 60 day closing price 
last_60_days = new_df1[-60:].values
#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#Create an empty list
X_test = []
#Append teh past 60 days
X_test.append(last_60_days_scaled)
#Convert the X_test data set to a numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Get the predicted scaled price
pred_price = model.predict(X_test)
#undo the scaling 
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)n


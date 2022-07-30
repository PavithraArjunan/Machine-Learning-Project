#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[4]:


data = pd.read_csv(r"C:\Users\Lenovo\Downloads\diabetes.csv")


# In[5]:


data.head()


# In[6]:


data.columns


# In[7]:


data.isnull().sum()


# In[8]:


data.shape


# In[9]:


data.describe()


# In[10]:


data['Outcome'].value_counts()


# In[11]:


data.groupby('Outcome').mean()


# In[12]:


# 0 --> non-diabetes
# 1 --> diabetes


# In[16]:


x = data.drop(['Outcome'],axis=1)
y = data['Outcome']
print(x)


# In[17]:


print(y)


# In[19]:


# Data Standardization
scaler = StandardScaler()
scaler.fit(x)


# In[20]:


standardized_data = scaler.transform(x)


# In[21]:


print(standardized_data)


# In[24]:


x = standardized_data
y = data['Outcome']
print(x)


# In[23]:


print(y)


# In[31]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)
print(x.shape,x_train.shape,x_test.shape)


# In[32]:


# Training The Model
model = svm.SVC(kernel='linear')


# In[33]:


# Training the support vector Machine Classifier
model.fit(x_train,y_train)


# In[40]:


# Model Evaluation - Accuracy Score
x_train_predict = model.predict(x_train)
train_accuracy_score = accuracy_score(x_train_predict,y_train)
print("Accuracy score for the training data:",train_accuracy_score)


# In[41]:


x_test_predict = model.predict(x_test)
test_accuracy_score = accuracy_score(y_test,x_test_predict)
print('Accuracy score for the test data',test_accuracy_score)


# In[48]:


# Making a Predictive System
input_data =(5,166,72,19,175,25.8,0.587,51)

# changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshaping the array as we are predicting for one instance
input_data = input_data_as_numpy_array.reshape(1,-1)

#standardize the input data
std_data = scaler.transform(input_data)
print(std_data)

prediction = model.predict(std_data)
print(prediction)

if(prediction[0]==0):
    print('(HEALTHY)--No diabetic')
else:
    print('DIABETIC PATIENT')


# In[51]:


input_data =(1,85,66,29,0,26.6,0.351,31)
input_data_as_numpy_array = np.asarray(input_data)
input_data = input_data_as_numpy_array.reshape(1,-1)
std_data = scaler.transform(input_data)
print(std_data)
prediction = model.predict(std_data)
print(prediction)

if(prediction[0]==0):
    print('(HEALTHY)--No diabetic')
else:
    print('DIABETIC PATIENT')


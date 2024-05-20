#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[2]:


diabetes_dataset = pd.read_csv('diabetes.csv') 


# In[3]:


diabetes_dataset.head()


# In[4]:


diabetes_dataset.shape


# In[5]:


# getting the statistical measures of the data
diabetes_dataset.describe()


# In[6]:


diabetes_dataset['Outcome'].value_counts()


# In[7]:


diabetes_dataset.groupby('Outcome').mean()


# In[8]:


# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']


# In[9]:


print(X)


# In[12]:


print(Y)


# # Data Standardization

# In[13]:


scaler = StandardScaler()


# In[14]:


scaler.fit(X)


# In[15]:


standardized_data = scaler.transform(X)


# In[16]:


print(standardized_data)


# In[17]:


X = standardized_data
Y = diabetes_dataset['Outcome']


# In[18]:


print(X)
print(Y)


# Train Test Split

# In[19]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)


# In[20]:


print(X.shape, X_train.shape, X_test.shape)


# Training the Model

# In[21]:


classifier = svm.SVC(kernel='linear')


# Training the support vector Machine Classifier

# In[22]:


classifier.fit(X_train, Y_train)


# 
# 
# Accuracy Score

# In[23]:


# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[24]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[25]:


# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[26]:


print('Accuracy score of the test data : ', test_data_accuracy)


# Making a Predictive System

# In[27]:


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[ ]:





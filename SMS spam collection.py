#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[14]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re


# In[15]:


df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)


# In[17]:


df.head()


# In[18]:


# Replace ham with 0 and spam with 1
df = df.replace(['ham','spam'],[0, 1]) 


# In[19]:


df.head()


# In[20]:


##Count the number of words in each Text
df['Count']=0
for i in np.arange(0,len(df.v2)):
    df.loc[i,'Count'] = len(df.loc[i,'v2'])


# In[21]:


df.head()


# In[22]:


# Total ham(0) and spam(1) messages
df['v1'].value_counts()


# In[23]:


df.info()


# In[24]:


corpus = []
ps = PorterStemmer()


# In[25]:


# Original Messages

print (df['v2'][0])
print (df['v2'][1])


# In[26]:


##Processing Messages


for i in range(0, 5572):

    # Applying Regular Expression
    
    '''
    Replace email addresses with 'emailaddr'
    Replace URLs with 'httpaddr'
    Replace money symbols with 'moneysymb'
    Replace phone numbers with 'phonenumbr'
    Replace numbers with 'numbr'
    '''
    msg = df['v2'][i]
    msg = re.sub('\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', df['v2'][i])
    msg = re.sub('(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', df['v2'][i])
    msg = re.sub('£|\$', 'moneysymb', df['v2'][i])
    msg = re.sub('\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', 'phonenumbr', df['v2'][i])
    msg = re.sub('\d+(\.\d+)?', 'numbr', df['v2'][i])
    
    ''' Remove all punctuations '''
    msg = re.sub('[^\w\d\s]', ' ', df['v2'][i])
    
    if i<2:
        print("\t\t\t\t MESSAGE ", i)
    
    if i<2:
        print("\n After Regular Expression - Message ", i, " : ", msg)
    
    # Each word to lower case
    msg = msg.lower()    
    if i<2:
        print("\n Lower case Message ", i, " : ", msg)
    
    # Splitting words to Tokenize
    msg = msg.split()    
    if i<2:
        print("\n After Splitting - Message ", i, " : ", msg)
    
    # Stemming with PorterStemmer handling Stop Words
    msg = [ps.stem(word) for word in msg if not word in set(stopwords.words('english'))]
    if i<2:
        print("\n After Stemming - Message ", i, " : ", msg)
    
    # preparing Messages with Remaining Tokens
    msg = ' '.join(msg)
    if i<2:
        print("\n Final Prepared - Message ", i, " : ", msg, "\n\n")
    
    # Preparing WordVector Corpus
    corpus.append(msg)


# In[27]:


cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()


# In[29]:


##Applying Classification
#Input : Prepared Sparse Matrix
#Ouput : Labels (Spam or Ham)

y = df['v1']
print (y.value_counts())

print(y[0])
print(y[1])


# In[37]:


#Encoding Labels
le = LabelEncoder()
y = le.fit_transform(y)

print(y[0])
print(y[1])


# In[40]:


##Splitting to Training and Testing DATA
xtrain, xtest, ytrain, ytest = train_test_split(x, y,test_size= 0.20, random_state = 0)


# In[41]:


##Applying Guassian Naive Bayes

bayes_classifier = GaussianNB()
bayes_classifier.fit(xtrain, ytrain)


# In[42]:


# Predicting
y_pred = bayes_classifier.predict(xtest)


# In[43]:


# Evaluating
cm = confusion_matrix(ytest, y_pred)
cm


# In[44]:


print ("Accuracy : %0.5f \n\n" % accuracy_score(ytest, bayes_classifier.predict(xtest)))
print (classification_report(ytest, bayes_classifier.predict(xtest)))


# In[45]:


##Applying Decision Tree
dt = DecisionTreeClassifier(random_state=50)
dt.fit(xtrain, ytrain)


# In[46]:


# Predicting
y_pred_dt = dt.predict(xtest)


# In[47]:


#Results
# Evaluating
cm = confusion_matrix(ytest, y_pred_dt)

print(cm)


# In[48]:


print ("Accuracy : %0.5f \n\n" % accuracy_score(ytest, dt.predict(xtest)))
print (classification_report(ytest, dt.predict(xtest)))


# In[ ]:


Final Accuracy
Decision Tree : 96.861%
Guassian NB : 87.085%


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
df=pd.read_csv("Cardiology.csv")
df.head()


# In[2]:


df=pd.read_csv("Cardiology.csv")


# In[3]:


df.info()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df[df.duplicated(keep=False)]


# In[7]:


sns.boxplot(x='age',data=df)


# In[8]:


sns.boxplot(x='trestbps', data=df)


# In[9]:


from matplotlib import pyplot
df.hist()
pyplot.show()


# In[10]:


feature_columns=['age', 'sex', 'trestbps',	'chol',	'fbs',	'restecg','thalach','exang','oldpeak','slope','ca'	,'thal' ]
X=df[feature_columns]
X


# In[11]:


y=df['target']
y


# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4,shuffle=True)


# In[13]:


X_train


# In[14]:


from sklearn.svm import SVC
def svm_model(X_train,y_train,X_test):
    svm = SVC(kernel='linear', C=1,random_state=0)
    svm.fit(X_train,y_train)
    y_pred=svm.predict(X_test)
    return y_pred


# In[15]:


y_pred=svm_model(X_train,y_train,X_test)
y_pred


# In[16]:


from sklearn.metrics import accuracy_score
accuracy_sklearn = accuracy_score(y_test, y_pred)*100
print('Model Accuracy:',accuracy_sklearn)


# In[17]:


from sklearn.linear_model import LogisticRegression
def LR_model(X_train,y_train,X_test):
    LR = LogisticRegression()
    LR.fit(X_train,y_train)
    y_pred=LR.predict(X_test)
    return y_pred


# In[18]:


y_pred=LR_model(X_train,y_train,X_test)
y_pred


# In[19]:


from sklearn.metrics import accuracy_score
accuracy_sklearn = accuracy_score(y_test, y_pred)*100
print('Model Accuracy:',accuracy_sklearn)


# In[20]:


from sklearn.neighbors import KNeighborsClassifier
def knn_model(X_train,y_train,X_test,k):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    return y_pred


# In[21]:


y_pred=knn_model(X_train,y_train,X_test,k=6)
y_pred


# In[22]:


from sklearn.metrics import accuracy_score
accuracy_sklearn = accuracy_score(y_test, y_pred)*100
print('Model Accuracy:',accuracy_sklearn)


# In[23]:


k_value=range(1,20)
accuracy=[]
for k in k_value:
    y_predict=knn_model(X_train,y_train,X_test,k)
    accur=accuracy_score(y_test,y_predict)
    accuracy.append(accur)


# In[24]:


plt.xlabel('k')
plt.ylabel('accuracy')
plt.plot(k_value,accuracy,c='g')
plt.show()


# In[ ]:





# In[ ]:





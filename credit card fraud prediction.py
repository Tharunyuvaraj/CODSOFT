#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import*
from sklearn.metrics import*
from sklearn.tree import*
from sklearn.model_selection import*
from sklearn.tree import*
from sklearn.ensemble import*


# In[6]:


d = pd.read_csv("creditcard.csv")
print(d.shape)


# In[7]:


d.head(10)


# In[8]:


d.tail()


# In[9]:


d.info()


# In[10]:


d.describe()


# In[11]:


def t10(s):
    
    # Get the top 10 amounts and their counts
    top = d[d["Class"] == s]["Amount"].value_counts().head(15)

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    top.plot(kind='bar', color='yellow')
    plt.title(f'Top 15 Amounts for Class {s} Transactions')
    plt.xlabel('Amount')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()


# In[15]:


t10(1)
t10(0)


# In[24]:


y=d['Class']
X=d.drop(['Class'], axis=1)

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2)

model=DecisionTreeClassifier()

model.fit(X_train,y_train)
yt=model.predict(X_test)
accuracy_score(y_test,yt)


# In[22]:


pd.crosstab(y_test,y_hat)


# In[28]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# In[ ]:





# In[ ]:





# In[ ]:





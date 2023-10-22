#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


d=pd.read_csv("IRIS.csv")
d.head()


# In[3]:


d.tail()


# In[4]:


d.info()


# In[8]:


d.describe(include="all")


# In[7]:


d.isnull().sum()


# In[14]:


plt.figure(figsize=(8,6));
sns.pairplot(d,kind='reg',hue ='species',palette="husl" );


# In[20]:


plt.figure(figsize=(10,6));
cmap = sns.cubehelix_palette(dark=.5, light=.4, as_cmap=False)
l = sns.scatterplot(x="petal_length", y="petal_width",hue="species",size="species",sizes=(20,100),legend="full",data=d);


# In[28]:


plt.figure(figsize=(10,6));
cmap = sns.cubehelix_palette(dark=.5, light=.4, as_cmap=False)
l = sns.histplot(x="sepal_length", y="sepal_width",hue="species",data=d,element="poly");


# In[30]:


from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
d['species'] = lb_make.fit_transform(d['species'])
d.sample(5)


# In[36]:


y = d.species
X = d.drop('species',axis = 1)


# In[37]:


from sklearn.model_selection import KFold,train_test_split,cross_val_score
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[38]:


#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,y_train)


# In[40]:


y_pred = knn.predict(X_test)
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[42]:


#navies bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,y_train)

GaussianNB(priors=None, var_smoothing=1e-09)

y_pred = nb.predict(X_test)



print('accuracy is',accuracy_score(y_pred,y_test))



# In[44]:


df = d[40:]
y = df.species
X = df.drop('species',axis = 1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)





# In[45]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)


# In[46]:


y_pred = lr.predict(X_test)


print('accuracy is',accuracy_score(y_pred,y_test))



# In[ ]:





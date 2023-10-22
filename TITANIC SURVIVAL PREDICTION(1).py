#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


d=pd.read_csv("TITANIC_DATASET.csv")
d.head(5) # displaying the head of the dataset


# In[4]:


d.tail() # displaying the tail of the dataset


# In[5]:


d.info() # displaying information of the dataset


# In[6]:


d.describe()


# In[10]:


d.isnull().sum() # sum of null values


# In[30]:


d['Age']=d['Age'].fillna(d['Age'].mean()).astype(int)  # Replacing the NULL as mean value
d['Fare']=d['Fare'].fillna(0)  # Replacing the NULL as 0 


# In[12]:


d.isnull().sum() # Checking the value


# In[14]:


d.dtypes #dataypes of each coloumn


# In[26]:


d.drop(columns=['PassengerId','Pclass','Name','SibSp','Cabin','Parch'],axis=0,inplace=True)
d.head(10)


# In[34]:


d["Fare"] = d.Fare.astype(int)
d.tail(15)


# In[35]:


d[d['Age']>60]


# In[39]:


d[(d['Sex']=='female') & (d['Survived']==1)]


# In[49]:


p= sns.countplot(data =d, x="Survived",hue="Survived", palette = ["blue", "black"])
p.set_xlabel("Survived")
p.set_ylabel("Count")
p.set_title("Survival Count Of Men & Women")
p.legend(title = "Legend", labels = ["Not Survived", "Survived"])
plt.xticks([0,1],["Male", "Female"])
plt.show()


# In[62]:


ec= d['Embarked'].value_counts()
plt.figure(figsize=(7, 6))
plt.pie(ec, labels=ec.index, autopct='%1.1f%%', colors=['yellow', '#99ff77', '#ff8688'])
plt.title("Graphical representation of Embarked")
plt.legend(["Q", "S", "C"])
plt.show()


# In[68]:


_,p = plt.subplots(figsize = (8, 6))
sns.countplot(data = d, x = "Embarked", hue = "Survived", palette=["blue", "yellow"])

p.set_xlabel("Embarked")
p.set_xticklabels(["Q", "S", "C"])
p.set_ylabel("Number of passengers")
p.legend(title = "Legends", labels = ["Not Survived", "Survived"])
plt.plot()


# In[ ]:





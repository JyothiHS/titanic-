#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv("C:\\Users\\Admin\\Downloads\\Titanic-Dataset.csv")


# In[5]:


df.head()


# ## Exploratory data analysis

# ### misssing data

# In[6]:


df.isnull()


# In[20]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[7]:


df.isnull().sum()


# In[9]:


sns.countplot(x='Survived',data =df, hue = 'Sex')


# In[16]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue ='Pclass',data = df)


# In[12]:


sns.distplot(df['Age'].dropna(),kde = False, color = 'red', bins = 10)


# In[13]:


sns.countplot(x='SibSp', data =df)


# In[18]:


df['Fare'].hist(color='green',bins= 40, figsize=(8,4))


# In[19]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y ='Age',data=df,palette='winter')


# In[21]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# In[23]:


df['Age'] = df[['Age','Pclass']].apply(impute_age,axis=1)


# In[25]:


df.drop('Cabin',axis=1,inplace=True)


# In[26]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[27]:


df.dropna(inplace=True)


# In[28]:


df.info()


# In[33]:


pd.get_dummies(df['Embarked'],drop_first=True).head()


# In[34]:



sex = pd.get_dummies(df['Sex'],drop_first=True)
embark = pd.get_dummies(df['Embarked'],drop_first=True)


# In[37]:


df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[39]:


df = pd.concat([df,sex,embark],axis=1)


# In[40]:


df.head()


# ### Logistic regression

# In[41]:


df.drop('Survived',axis=1).head()


# In[42]:


df['Survived'].head()


# In[46]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[44]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived',axis=1), 
                                                    df['Survived'], test_size=0.30, 
                                                    random_state=101)


# In[47]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[48]:


LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)


# In[49]:


predictions = logmodel.predict(X_test)


# In[50]:


from sklearn.metrics import confusion_matrix


# In[51]:


accuracy=confusion_matrix(y_test,predictions)


# In[52]:


accuracy


# In[53]:


from sklearn.metrics import accuracy_score


# In[54]:


accuracy=accuracy_score(y_test,predictions)
accuracy


# In[55]:


predictions


# In[56]:


from sklearn.metrics import classification_report


# In[57]:


print(classification_report(y_test,predictions))


# In[ ]:





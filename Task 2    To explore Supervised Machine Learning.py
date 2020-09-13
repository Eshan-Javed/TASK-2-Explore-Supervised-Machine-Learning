#!/usr/bin/env python
# coding: utf-8

# # MUHAMMAD ESHAN JAVED

# In[ ]:





# # Task 2 Linear Regression with Python Scikit Learn
# In this section we will see how the Python Scikit-Learn library for machine learning can be used to implement regression functions. We will start with simple linear regression involving two variables.
# 

# # Simple Linear Regression
# 
# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

# In[32]:


# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading data from the link

# In[33]:


# Importing the dataset
data_path="http://bit.ly/w-data"
data=pd.read_csv(data_path)


# In[34]:


data.head(10)


# In[35]:


#Import Info
data.info()


# In[36]:


# Check missing values
data.isnull().sum()


# In[37]:


data.describe()


# # Plotting 'HOURS STUDIED' VS 'PERCENTAGE'

# In[38]:


# visualizing the data
data.plot(x='Hours',y='Scores',style='r.')
plt.title('Initial Chart')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage')
plt.grid()
plt.show()


# In[39]:


# Plotting regression line to  see fitting
sns.regplot(x='Hours',y='Scores',data=data, color='red')
plt.title('Initial Chart')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage')
plt.grid()
plt.show()


# In[40]:


# Data preparation
x=data[['Hours']].values
y=data[['Scores']].values


# In[41]:


# Importing required function and splitting the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)


# # Linear Regression Model on Training Set

# In[42]:


# Training phase
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train,y_train)


# In[43]:


# Obtaining  coefficient and intercept of the model
inter = regression.intercept_
coeff = regression.coef_ 
print('The Intercept is : ',inter)
print('The Coefficient is : ',coeff[0])


# In[44]:


# Making prediction
y_pred = regression.predict(x_test)


# In[45]:


# Visualize for training data
plt.scatter(x_train,y_train, label='score')
plt.plot(x_train, regression.predict(x_train),color='red', label='Line of best fit')
plt.xlabel('Hours Study')
plt.ylabel('Percentage')
plt.title('Best fit line on training data')
plt.legend()
plt.show()


# In[46]:


# Visualize for testing data
plt.scatter(x_test,y_test, label='score')
plt.plot(x_test, regression.predict(x_test),color='red', label='Line of best fit')
plt.xlabel('Hours Study')
plt.ylabel('Percentage')
plt.title('Best fit line on training data')
plt.legend()
plt.show()


# # Making Prediction

# In[47]:


# Making required predicition
print('No of Hours =',9.25)
print('Predicted score in % =',regression.predict([[9.25]])[0][0])


# In[48]:


# Evaluation of the model
from sklearn import metrics
print('Mean Absolute Error :',metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error :',metrics.mean_squared_error(y_test,y_pred))
print('Root mean Squared error :',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Objective
# #Identify the Which Variable have more impact on Body Fat
# 
# #To build the Model which will help to Predict the Body Fat  based on Various parameter ( independent Variable )

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[2]:


Body_fat_data = pd.read_csv(r"D:/PGDM/Sem 3/Machine Learning/Project/Linear Regression/bodyfat.csv")
Body_fat_data.info()


# In[3]:


Body_fat_data.head(5)


# # EDA
# #Missing Value

# In[4]:


Body_fat_data.isna().sum()  # no missing data


# # Box PLot

# # Outlier

# In[5]:


import seaborn as sns
sns.boxplot(data=Body_fat_data)


# # Treatment of Outlier -- Winsorizing Technique

# In[6]:


for i in Body_fat_data:                  # i = column name 
    if Body_fat_data[i].dtypes in ("float64","int64"): # Body_fat_data[i] = every column will select float & int column only
        q1 = Body_fat_data[i].quantile(0.25)  # for 1 column compute q1
        q3 = Body_fat_data[i].quantile(0.75)  # for 1 column compute q3
        iqr = q3-q1  # for 1 column compute IQR
        ul = q3 + 1.5*iqr # for 1 column compute UPPER LIMIT
        ll = q1 - 1.5*iqr # for 1 column compute LOWER LIMIT 
        Body_fat_data[i].clip(lower=ll,upper=ul,inplace=True)


# In[7]:


import seaborn as sns
sns.boxplot(data=Body_fat_data)


# In[8]:


import seaborn as sns
sns.boxplot(data=Body_fat_data)


# # Heat Map

# In[42]:


import matplotlib.pyplot as plt
corr_matrix = Body_fat_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.xlabel('Features')
plt.ylabel('Features')
plt.show()


# # Distribution Plot

# In[44]:


plt.figure(figsize=(10, 6))
sns.histplot(Body_fat_data['BodyFat'], kde=True, bins=20)
plt.title('Distribution of BodyFat')
plt.show()


# # Data Partition

# In[10]:


X = Body_fat_data.drop('BodyFat', axis= 1)
y = Body_fat_data[['BodyFat']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7 ,random_state=136)


# In[11]:


train = pd.concat([y_train, X_train], axis=1)
train.head()


# # Correlation

# In[12]:


Correlation = train.corr()
Correlation.style.applymap(lambda x: 'background-color : red'if x > 0.7 else '')


# #Correlation
# #Chest,Abdomen,Biceps, Wrist have high Impact on Body Fact
# #Multi-Colinearity is Present

# # VIF

# In[13]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
X1 = Body_fat_data.drop(['BodyFat'],axis=1)
series_before = pd.Series([variance_inflation_factor(X1.values, i) 
                           for i in range(X1.shape[1])],  # i=0,1,2,...8
                          index=X1.columns)  # column name
series_before

#Multi- colinearity is Present in Dataset
# # Model Building
# #Method 2 -- Variable selection Method
# #Forward Selection Method
# #Backward Selection Method

# In[14]:


from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector as sfs

lreg = LinearRegression()
Model = sfs(lreg, n_features_to_select = 5, direction='forward', scoring='r2')
Model.fit(X_train,y_train)


# In[15]:


Model.feature_names_in_


# In[16]:


Model.get_feature_names_out()


# # Method 2 -- sklearn

# In[17]:


from sklearn.linear_model import LinearRegression


# In[18]:


X_train = X_train.loc[:,[ 'Density', 'Age', 'Weight', 'Height', 'Neck']]
X_train.head()


# In[19]:


train = pd.concat([X_train,y_train], axis=1)
train.head()


# In[20]:


Model3 = LinearRegression()
Model3.fit(X_train,y_train)


# In[21]:


np.round(Model3.intercept_,4)


# In[22]:


np.round(Model3.coef_,3)


# # Converting scientific notation  to normal Decimal 

# In[23]:


import numpy as np


# In[24]:


arr = np.array([[-4.01547e+02,  2.80000e-02,  2.90000e-02, -4.00000e-02,-6.10000e-02]])
np.set_printoptions(suppress=True)


# In[25]:


print(arr)


# In[26]:


Model3.feature_names_in_


# # Model
# 
# #y= 441.728-401.547(Density)+ 0.028(Age)+0.029(Weight)-0.04(Height)-0.061(Neck)

# In[27]:


#Prediction of train


# In[28]:


import numpy as np
train['fitted_value'] = np.round(Model3.predict(X_train),2)
train['Residual'] = np.round(train.BodyFat - train.fitted_value,2)
train.head()


# In[29]:


from sklearn.metrics import r2_score
r2 = r2_score(train.BodyFat,train.fitted_value)
print('R-Squared score for model Performance on Train : ', np.round(r2,2)*100)


# # Assumption of Linear Regression

# In[30]:


Body_fat_data.plot.scatter(x='Weight', y='BodyFat', title='BodyFat vs Wieght')


# In[31]:


#Homoscedasicity


# In[32]:


sns.scatterplot(x='fitted_value', y='Residual',data=train)


# In[33]:


import statsmodels.api as sm
from matplotlib import pyplot as plt

fig = sm.qqplot(train['Residual'], fit=True, line='s') # s indicate standardized line
plt.show()


# In[34]:


train['Residual'].plot.hist()


# #Prediction on Test

# In[35]:


X_test = X_test.loc[:,['Density', 'Age', 'Weight', 'Height', 'Neck']]
X_test.head()


# In[36]:


test = pd.concat([X_test,y_test], axis=1)
test.head()


# In[37]:


import numpy as np
test['Prediction'] = np.round(Model3.predict(X_test),2)
test['Error / Residual'] = np.round(test.BodyFat - test.Prediction,2)
test.head()


# # Comparision

# In[38]:


from sklearn.metrics import r2_score
r2 = r2_score(train.BodyFat,train.fitted_value)
print('R-Squared score for model Performance on Train : ', np.round(r2,2)*100)


# In[39]:


r2 = r2_score(test.BodyFat,test.Prediction)
print('R-Squared score for model Performance on Test : ', np.round(r2,2)*100)


# # Model Perfomance
# #Model prediction accuracy is 97%, that means model is overfitted

# # Loss Function -- RMSE

# In[40]:


from sklearn.metrics import mean_squared_error 

model_mse = mean_squared_error(train['BodyFat'],train['fitted_value'])
model_rmse = np.sqrt(model_mse)

print("RMSE of Train Data : ",np.round(model_rmse,2)) 


# In[41]:


model_mse = mean_squared_error(test['BodyFat'],test['Prediction'])
model_rmse = np.sqrt(model_mse)

print("RMSE of Test Data : ",np.round(model_rmse,2)) #21


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # BIG_mart_sale_prediction
# 

# In[131]:


#importing libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.linear_model import  Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor


# In[132]:


train_data=pd.read_csv(r"C:\Users\AMIT KUMAR SINGH\Desktop\ML_projects\training_data.csv")
test_data=pd.read_csv(r"C:\Users\AMIT KUMAR SINGH\Desktop\ML_projects\test_data.csv")
train_data['source']='train'
test_data['source']='test'
df = pd.concat([train_data, test_data],ignore_index=True)


# In[133]:


#Check missing values:
df.apply(lambda x: sum(x.isnull()))
#Replacing the nan values with its mean value
df=df.fillna(df.mean())
df["Outlet_Size"].fillna(method="ffill",inplace=True)
#data.apply(lambda x: sum(x.isnull()))


# In[134]:


#Now we have successfully cleaned our complete dataset.
df["Item_Fat_Content"].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'},inplace=True)
#data["Item_Fat_Content"].value_counts()


# In[135]:


#Item type combine:
df['Item_Identifier'].value_counts()
df['Item_Type_Combined'] = df['Item_Identifier'].apply(lambda x: x[0:2])
df['Item_Type_Combined'] = df['Item_Type_Combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})
#data['Item_Type_Combined'].value_counts()


# In[136]:


le = LabelEncoder()
#New variable for outlet
df['Outlet'] = le.fit_transform(df['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Item_Type_Combined','Outlet_Size','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])


# In[137]:


#One Hot Coding:
df= pd.get_dummies(df, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                                     'Item_Type_Combined','Outlet'])


# In[138]:


#Drop the columns which have been converted to different types:
df.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)
#Divide into test and train:
train = df.loc[df['source']=="train"]
test = df.loc[df['source']=="test"]
#Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)


# In[139]:


#Export files as modified versions:
train.to_csv(r"C:\Users\AMIT KUMAR SINGH\Desktop\ML_projects\train1_modified.csv",index=False)
test.to_csv(r"C:\Users\AMIT KUMAR SINGH\Desktop\ML_projects\test1_modified.csv",index=False)


# In[140]:


# Reading modified data
train1 = pd.read_csv(r"C:\Users\AMIT KUMAR SINGH\Desktop\ML_projects\train1_modified.csv")
test1= pd.read_csv(r"C:\Users\AMIT KUMAR SINGH\Desktop\ML_projects\test1_modified.csv")


# In[141]:


x_train = train1.drop(['Item_Outlet_Sales', 'Outlet_Identifier','Item_Identifier'], axis=1)
y_train = train1.Item_Outlet_Sales
x_test = test1.drop(['Outlet_Identifier','Item_Identifier'], axis=1)
Y_train=y_train[0:5681]


# # Sale Prediction Using Ridge Regression

# In[142]:


theta=np.zeros(len(x_train.columns))
alpha=0.0001
lmda=0.01
n_iter=400
for i in range(0,n_iter):
    
    h=(np.dot(x_train,theta))
    theta[0]=theta[0]-((alpha*(np.sum(h-y_train)))/len(x_train))
    for j in range(1,len(theta)):
        theta[j]=theta[j]*((1-(alpha*lmda))-(alpha*(np.sum(np.dot((h-y_train),x_train.iloc[:,j])))))/108

y_predR=(np.dot(x_test,theta))


# In[143]:


# plotting residual errors in training data 
plt.scatter((np.dot(x_train,theta)),(np.dot(x_train,theta)) - y_train,color = "red", s = .5, 
            label = 'Train data',marker = '.')
plt.scatter((np.dot(x_test,theta)),(np.dot(x_test,theta)) - y_predR,color = "blue",s = .05,
            label = 'Test data',marker = '.')
plt.xlabel('Attributes') 
plt.ylabel('Sale') 
# Measuring Accuracy
y_predR=(np.dot(x_test,theta))
errors=[]
for i in range(Y_train.shape[0]):
    error=(Y_train[i]-y_predR[i])/Y_train[i]
    errors.append(error)
errors=np.array(errors)
accuracy=100-error_mean
print("linear accuracy of model:",accuracy)
error_mean=np.mean(abs(errors))
RMSE=np.sqrt(metrics.mean_squared_error(Y_train, y_predR))
print("RMSE of model:", RMSE)


# In[144]:


alg2 = Ridge(alpha=0.01,normalize=True)
alg2.fit(x_train, y_train)
coefR = pd.Series(alg2.coef_,x_train.columns).sort_values()
coefR.plot(kind='bar', title='Model Coefficients') 
plt.xlabel('Attributes') 
plt.ylabel('Sale')


# # Sale Prediction Using Decision tree

# In[145]:


# Fitting Decision Tree Regression to the dataset
regressor = DecisionTreeRegressor(max_depth=15,min_samples_leaf=300)
regressor.fit(x_train, y_train)
# Predicting the test set results
y_pred2 = regressor.predict(x_test)
Y_train=y_train[:5681]
coef3 = pd.Series(regressor.feature_importances_,x_train.columns).sort_values(ascending=False)
coef3.plot(kind='bar', title='Feature Importances')
plt.xlabel('Attributes') 
plt.ylabel('Sale') 


# In[146]:


# plotting residual errors in training data 
plt.scatter(regressor.predict(x_train), regressor.predict(x_train) - y_train,color = "red", s = .5, 
            label = 'Train data',marker = '.')
plt.scatter(regressor.predict(x_test), regressor.predict(x_test) - y_pred2,color = "blue", s = 20,
            label = 'Test data',marker = '.')
plt.xlabel('Attributes') 
plt.ylabel('Sale')
tree_accuracy = round(regressor.score(x_train,y_train)*100,20)
print("tree accuracy of model:", tree_accuracy)
RMSE=np.sqrt(metrics.mean_squared_error(Y_train, y_pred2))
print("RMSE of model:", RMSE)


# # Sale Prediction Using Random Forest

# In[147]:


# Fitting Random Forest Regression to the dataset
regressor = RandomForestRegressor(n_estimators=100,max_depth=6, min_samples_leaf=50,n_jobs=4)
regressor.fit(x_train, y_train)
# Predicting the test set results
y_pred3 = regressor.predict(x_test)
coef6 = pd.Series(regressor.feature_importances_,x_train.columns).sort_values(ascending=False)
coef6.plot(kind='bar', title='Feature Importances') 
plt.xlabel('Attributes') 
plt.ylabel('Sale') 


# In[148]:


# plotting residual errors in training data 
plt.scatter(regressor.predict(x_train), regressor.predict(x_train) - y_train,color = "red", s = .5, 
            label = 'Train data',marker = '.')
plt.scatter(regressor.predict(x_test), regressor.predict(x_test) - y_pred3,color = "blue", s = .5,
            label = 'Test data',marker = '.')
plt.xlabel('Attribute') 
plt.ylabel('Sale') 
Raf_accuracy = round(regressor.score(x_train,y_train)*100,10)
print("Random forest accuracy of model:", Raf_accuracy)
RMSE=np.sqrt(metrics.mean_squared_error(Y_train, y_pred3))
print("RMSE of model:", RMSE)


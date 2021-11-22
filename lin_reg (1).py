# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 10:46:07 2021

@author: Izat
"""



import numpy as np # data manipulation library
from sklearn.linear_model import LinearRegression
import pandas as pd # data manipulation library
import seaborn as sns # data visualization lib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#data loading / implementation
data = pd.read_csv(r"C:\Users\ASUS VivaBook\Desktop\PYTHON QSS\USA_Housing.csv")
data.head()
data.info()
statistics = data.describe()
statistics
cols = data.columns
cols

#data exploration
sns.pairplot(data) # visual observation
sns.heatmap(data.corr(), annot=True)
data.isnull().sum()


#data preprocessing
#data[col].fillna(data[col].mean(),inplace=True)
#data[col].fillna(0,inplace=True)
#data= data.dropna(axis=1)
#data= data.dropna(axis=1,thresh=2)


#replace outliers
def outlier_treatment(datacolumn):
    sorted(datacolumn)
    Q1,Q3 = np.percentile(datacolumn , [25,75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range,upper_range

for col in data.select_dtypes(exclude="object").columns:
    
    lowerbound,upperbound = outlier_treatment(data[col])
    df\ata[col]=np.clip(data[col],a_min=lowerbound,a_max=upperbound) # datacolumn, lowerbound,upperbound
    


#delete outliers
for i in data.columns:
    if i=='Sales':
        continue    
    q3=np.quantile(data[i], 0.75)    
    q1=np.quantile(data[i], 0.25)    
    iqr=1.5*(q3-q1)    
    b0=q1-iqr    
    b1=q3+iqr    
    data=data[(b0<data[i]) & (data[i]<b1)]




# data splittin (X, y)/  preparation
X = data[[cols[0], cols[1], cols[2], cols[3], cols[4]]]
y = data['Price']

#training/validation split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#model building (training process)
lin_reg = LinearRegression (normalize = True)
lin_reg.fit(X_train, y_train) #training line / model building line
betta0 = lin_reg.intercept_
coefficients = lin_reg.coef_
betta0
coefficients
#estimation of the productivity of the model / evaluation of the model

y_pred = lin_reg.predict(X_test)

plt.plot(prediction, y_test, 'o')
plt.xlabel('Predcition')
plt.ylabel('Real')

sns.boxplot(x="Price", y="Avg. Area Income", data=data,palette='Blues')

from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures

R2 = metrics.r2_score(y_test, y_pred)
print('R^2:' , R2)
n = X_test.shape[0] #sample size
p = X_test.shape[1] #number of predictors
print('Adjusted R^2 :' , 1-(1-R2)*(n-1)/(n-p-1))
print('Adjusted R^2 :'  , 1 - metrics.r2_score(y_test, y_pred))
print('Mean Absolute Error:' , metrics.mean_absolute_error(y_test, y_pred)) 
print('Mean Squared Error:' , metrics.mean_squared_error(y_test, y_pred)) 
print('Root Mean Squared Error:' , np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#polynomial regression
poly_reg = PolynomialFeatures(degree=2)

X_train_2d = poly_reg.fit_transform(X_train)
X_test_2d = poly_reg.fit_transform(X_test)

lin_reg.fit(X_train_2d,y_train)

y_pred = lin_reg.predict(X_test_2d)
plt.plot(y_test, y_pred, 'o')

R2 = metrics.r2_score(y_test, y_pred)
print('R^2:' , R2)
n = X_test.shape[0] #sample size
p = X_test.shape[1] #number of predictors
print('Adjusted R^2 :' , 1-(1-R2)*(n-1)/(n-p-1))
print('Adjusted R^2 :'  , 1 - metrics.r2_score(y_test, y_pred))
print('Mean Absolute Error:' , metrics.mean_absolute_error(y_test, y_pred)) 
print('Mean Squared Error:' , metrics.mean_squared_error(y_test, y_pred)) 
print('Root Mean Squared Error:' , np.sqrt(metrics.mean_squared_error(y_test, y_pred)))















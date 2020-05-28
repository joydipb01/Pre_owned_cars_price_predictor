# -*- coding: utf-8 -*-
"""
Created on Tue May 26 11:28:21 2020

@author: Joydip
"""

import os
import pandas as pd
import numpy as np
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

sb.set(rc={'figure.figsize':(11.7, 8.27)}) #setting dimensions for the plot

os.chdir('C:/Users/Joydip/Documents')
cars=pd.read_csv('cars_sampled.csv')
cars2=cars.copy() #deep copy of the original dataset

#Knowing the structure of the dataset, and summarising it:
cars2.info()

pd.set_option('display.float_format', lambda x: "%.3f"%x)
pd.set_option('display.max_columns', 500)
cars2.describe()

'''
"price" is extremely skewed. "postalCode" is not required for the analysis
'''

#Removing unwanted columns
col=['name', 'dateCrawled', 'dateCreated', 'postalCode', 'lastSeen']
cars2=cars2.drop(columns=col, axis=1)

#Removing duplicate records
cars2.drop_duplicates(keep='first', inplace=True)

#Data Cleaning

#Determining number of missing values in each column
cars2.isnull().sum()

#Anlaysing the column "yearOfRegistration"
year_count=cars2['yearOfRegistration'].value_counts().sort_index()
sum(cars2['yearOfRegistration'] > 2018)
sum(cars2['yearOfRegistration'] < 1950)
sb.regplot(x='yearOfRegistration', y='price', scatter='True',
           fit_reg=False, data=cars2)
#For our anaysis, the working range for this column is 1950-2018

#Analysing the column "price"
price_count=cars2['price'].value_counts().sort_index()
sb.distplot(cars2['price'])
'''
Most, if not all, of the entries in the dataset are priced at 0
'''
cars2['price'].describe()
sb.boxplot(y=cars2['price'])
'''
Considerable outliers make the given column skewed. The behaviour of the 
variable cannot be ascertained stochastically.
'''
sum(cars2['price'] > 150000)
sum(cars2['price'] < 100)
#For our analysis, the working range for this column is 100-150000

#Analysing the column "powerPS"
power_count=cars2['powerPS'].value_counts().sort_index()
sb.distplot(cars2['powerPS'])
cars2['powerPS'].describe()
sb.boxplot(y=cars2['powerPS'])
'''
A problem similar to the "price" column: most entries in the dataset have 0
power and many outliers. However, this column is not as skewed as the "price"
column.
'''
sb.regplot(x='powerPS', y='price', scatter=True,
           fit_reg=False, data=cars2)
sum(cars2['powerPS'] > 500)
sum(cars2['powerPS'] < 10)
#For our analysis, the woring range for this column is 10-500

#Incorporating the said working ranges in the dataset
cars2=cars2[
        (cars2.yearOfRegistration<=2018)
        & (cars2.yearOfRegistration>=1950)
        & (cars2.price>=100)
        & (cars2.price<=150000)
        & (cars2.powerPS>=10)
        & (cars2.powerPS<=500)]

'''
Further simplification can be done by reducing the number of variables in the
dataframe. This can be done by creating a new column "age" by combining the
columns "yearOfRegistration" and "monthOfRegistration".
'''
cars2['monthOfRegistration']/=12
cars2['age']=(2018-cars2['yearOfRegistration'])+cars2['monthOfRegistration']
cars2['age']=round(cars2['age'], 2)
cars2['age'].describe()
cars2=cars2.drop(columns=['yearOfRegistration', 'monthOfRegistration'], axis=1)

#Visualising various parameters

#age
sb.distplot(cars2['age'])
sb.boxplot(y=cars2['age'])

#price
sb.distplot(cars2['price'])
sb.boxplot(y=cars2['price'])

#powerPS
sb.distplot(cars2['powerPS'])
sb.boxplot(y=cars2['powerPS'])

'''
Visualising the dependence of these parameters under the stipulated working
ranges
'''

#age-price
sb.regplot(x='age', y='price', scatter=True,
           fit_reg=False, data=cars2)
'''
Clearly, cars priced higher are fairly newer. As the car ages, price decreases.
However, there are some outliers to this trend(vintage cars).
'''

#powerPS-price
sb.regplot(x='powerPS', y='price', scatter=True,
           fit_reg=False, data=cars2)
'''
Higher priced cars have a higher power.
'''

#Working with categorical variables

#Column "seller"
cars2['seller'].value_counts()
pd.crosstab(cars2['seller'], columns='count', normalize=True)
sb.countplot(x='seller', data=cars2)
'''
Commercial sellers are not adequatly represented in the dataset for our 
analysis. Infact, there are an insignificant number of commercial sellers.
'''

#Column "offerType"
cars2['offerType'].value_counts()
sb.countplot(x='offerType', data=cars2)
'''
All cars are of the offerType 'offer' (one kind of offerType). Hence, this
column will not be of much use in our analysis.
'''

#Column "abtest"
cars2['abtest'].value_counts()
pd.crosstab(cars2['abtest'], columns='count', normalize=True)
sb.countplot(x='abtest', data=cars2)
'''
This column is, more or less, evenly distributed, with "test" having a larger 
count than "control"
'''
sb.boxplot(x='abtest', y='price', data=cars2)
'''
Both "test" and "control" have an almost equal distribution in price. Hence, 
this column does not have a telling effect on price.
'''

#Column "vehicleType"
cars2['vehicleType'].value_counts()
pd.crosstab(cars2['vehicleType'], columns='count', normalize=True)
sb.countplot(x='vehicleType', data=cars2)
sb.boxplot(x='vehicleType', y='price', data=cars2)
'''
From the visualisations, it is clear that vehicles "limousine", "small car"
and "station wagon" ahve the highest frequency. On the whole, this column
has an effect on price, as is noticeable from the various price ranges for 
different vehicles.
'''

#Column "gearbox"
cars['gearbox'].value_counts()
pd.crosstab(cars2['gearbox'], columns='count', normalize=True)
sb.countplot(x='gearbox', data=cars2)
sb.boxplot(x='gearbox', y='price', data=cars2)
'''
This column, too, has an adverse effect on price, as shown in the
visualisations
'''

#Column "model"
cars2['model'].value_counts()
pd.crosstab(cars2['model'], columns='count', normalize=True)
sb.countplot(x='model', data=cars2)
sb.boxplot(x='model', y='price', data=cars2)
'''
It is clear that the cars are distributed over many models. Hence, this column
is considered for our analysis.
'''

#Column "kilometer"
cars2['kilometer'].value_counts().sort_index()
pd.crosstab(cars2['kilometer'], columns='count', normalize=True)
sb.boxplot(x='kilometer', y='price', data=cars2)
cars2['kilometer'].describe()
sb.distplot(cars2['kilometer'], bins=8, kde=False)
sb.regplot(x='kilometer', y='price', scatter=True,
           fit_reg=False, data=cars2)
'''
In general, as the cumulative distance travelled by the car increases, the 
price drops
'''

#Column "fuelType"
cars2['fuelType'].value_counts()
pd.crosstab(cars2['fuelType'], columns='count', normalize=True)
sb.countplot(x='fuelType', data=cars2)
sb.boxplot(x='fuelType', y='price', data=cars2)
'''
"fuelType" affects the "price" of the car. Hence, this column is retained for 
our analysis.
'''

#Column "brand"
cars2['brand'].value_counts()
pd.crosstab(cars2['brand'], columns='count', normalize=True)
sb.countplot(x='brand', data=cars2)
sb.boxplot(x='brand', y='price', data=cars2)
'''
Cars are distributed over many brands, hence the column is included for our
analysis
'''

#column "notRepairedDamage"
cars2['notRepairedDamage'].value_counts()
pd.crosstab(cars2['notRepairedDamage'], columns='count', normalize=True)
sb.countplot(x='notRepairedDamage', data=cars2)
sb.boxplot(x='notRepairedDamage', y='price', data=cars2)
'''
The cars which have not been repaired (the 'yes' category in the column) are
priced lower. The column is included in our analysis.
'''

#Removing the insignificant columns

col1=['seller', 'offerType', 'abtest']
cars2=cars2.drop(col1, axis=1)
cars3=cars2.copy()

#Correlation between numerical columns

cars_select=cars2.select_dtypes(exclude=[object])
correlation=cars_select.corr()
round(correlation, 3)
cars_select.corr().loc[:, 'price'].abs().sort_values(ascending=False)[1:]
#"powerPS" has a higher correlation than the rest

'''
Linear Regression/Random Forest model based on:
    1. Data without missing values
    2. Data with imputed null values
'''

#Omitting missing values

cars_nonull=cars2.dropna(axis=0)

#Converting categorical variables into dummy variables
cars_nonull=pd.get_dummies(cars_nonull, drop_first=True)

#Building the model with the "cars_nonull" dataframe

#Separating input and output features
x1=cars_nonull.drop(['price'], axis='columns', inplace=False)
y1=cars_nonull['price']

#Plotting the column "price"
price=pd.DataFrame({"1. Before":y1, "2. After":np.log(y1)})
price.hist()
#It is prudent to continue using the logarithmic model of "price"

#Transforming price as a logarithmic entity
y1=np.log(y1)

#Splitting this data into training and test data
x_train, x_test, y_train, y_test=train_test_split(x1, y1, test_size=0.3,
                                                  random_state=3)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

#Creating a baseline model

'''
This is done for setting a benchmark and compare the results with the 
regression model, and is done using mean of the test data.
'''

#Finding mean of the test data
base_pred=np.mean(y_test)
print(base_pred)

#Repeating the same value till the length of the dataframe
base_pred=np.repeat(base_pred, len(y_test))

#Determining the root mean squared error
base_rmse=np.sqrt(mean_squared_error(y_test, base_pred))
print(base_rmse)
#Any other model which has an RMSE less than this will be taken as the final
#model

#Linear Regression with "cars_nonull" dataframe

#Setting intercept as true
lr=LinearRegression(fit_intercept=True)

#Model
model_lr=lr.fit(x_train, y_train)

#Predicting model on test set
cars_pred_lr=lr.predict(x_test)

#Computing mean and root mean squared errors
lr_mse=mean_squared_error(y_test, cars_pred_lr)
lr_rmse=np.sqrt(lr_mse)
print(lr_rmse)

#Computing the R-squared value
r2_lr_test=model_lr.score(x_test, y_test)
r2_lr_train=model_lr.score(x_train, y_train)
print(r2_lr_test, r2_lr_train)

#Regression diagnostics - Residual plot analysis
residuals=y_test-cars_pred_lr
sb.regplot(x=cars_pred_lr, y=residuals, scatter=True,
           fit_reg=False, data=cars2)
residuals.describe()
'''
A good model has been achieved with a lower RMSE value and residuals 
concentrated at zero
'''

#Random Forest with "cars_nonull"

#Model parameters
rf=RandomForestRegressor(n_estimators=100, max_features='auto', 
                         max_depth=100, min_samples_split=10,
                         min_samples_leaf=4, random_state=1)

#Model
model_rf=rf.fit(x_train, y_train)

#Predicting model on test set
cars_pred_rf=rf.predict(x_test)

#Computing mean and root mean squared errors
rf_mse=mean_squared_error(y_test, cars_pred_rf)
rf_rmse=np.sqrt(rf_mse)
print(rf_rmse)

#Computing the R-squared value
r2_rf_test=model_rf.score(x_test, y_test)
r2_rf_train=model_rf.score(x_train, y_train)
print(r2_rf_test, r2_rf_train)
'''
Clearly, the Random Forest model is performing a lot better than the Linear
Regression model due to a lesser RMSE and R-squared value
'''

#Imputing Data

cars_imputed=cars2.apply(lambda x:x.fillna(x.median()) \
                         if x.dtype=='float' else \
                         x.fillna(x.value_counts().index[0]))
cars_imputed.isnull().sum()

#Converting categorical variables to dummy variables
cars_imputed=pd.get_dummies(cars_imputed, drop_first=True)

#Model building

#Separating input and output features
x2=cars_imputed.drop(['price'], axis='columns', inplace=False)
y2=cars_imputed['price']

#Plotting the column "price"
prices=pd.DataFrame({"1. Before:":y2, "2. After:":np.log(y2)})
prices.hist()

#Transforming price as a logarithmic function
y2=np.log(y2)

#Splitting data into test and train data
x_train1, x_test1, y_train1, y_test1=train_test_split(x2, y2, test_size=0.3,
                                                      random_state=3)
print(x_train1.shape, x_test1.shape, y_train1.shape, y_test1.shape)

#Baseline model

#Finding mean for the test data
base_pred1=np.mean(y_test1)
print(base_pred1)

#Repeating the same value untill the length of the dataset
base_pred1=np.repeat(base_pred1, len(y_test1))

#Finding the RMSE
base_rmse1=np.sqrt(mean_squared_error(y_test1, base_pred1))
print(base_rmse1)

#Linear Regression with "cars_imputed"

#Setting intercept as True
lr1=LinearRegression(fit_intercept=True)

#Model
model_lr1=lr1.fit(x_train1, y_train1)

#Predicting model on test set
cars_pred_lr1=lr1.predict(x_test1)

#Computing MSE and RMSE
lr_mse1=mean_squared_error(y_test1, cars_pred_lr1)
lr_rmse1=np.sqrt(lr_mse1)
print(lr_rmse1)

#Computing the R-squared value
r2_lr1_test=model_lr1.score(x_test1, y_test1)
r2_lr1_train=model_lr1.score(x_train1, y_train1)
print(r2_lr1_test, r2_lr1_train)

#Regression diagnostics - Residual plot analysis
residuals1=y_test1-cars_pred_lr1
sb.regplot(x=cars_pred_lr1, y=residuals1, scatter=True,
           fit_reg=False, data=cars2)
residuals1.describe()

#Random Forest with "cars_imputed"

#Model parameters
rf1=RandomForestRegressor(n_estimators=100, max_features='auto', 
                         max_depth=100, min_samples_split=10,
                         min_samples_leaf=4, random_state=1)

#Model
model_rf1=rf1.fit(x_train1, y_train1)

#Predicting model on test set
cars_pred_rf1=rf1.predict(x_test1)

#Computing mean and root mean squared errors
rf1_mse=mean_squared_error(y_test1, cars_pred_rf1)
rf1_rmse=np.sqrt(rf1_mse)
print(rf1_rmse)

#Computing the R-squared value
r2_rf1_test=model_rf1.score(x_test1, y_test1)
r2_rf1_train=model_rf1.score(x_train1, y_train1)
print(r2_rf1_test, r2_rf1_train)
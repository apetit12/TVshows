#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 17:35:50 2019

IBM: DA0101 - Analyzing Data with Python
Example of data analysis using Python with a vehicle specifications database

@author: antoinepetit
"""
###############################################################################
#                               Import libraries
###############################################################################

import pandas as pd
import csv
import urllib2
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

###############################################################################
#                                  Import data
###############################################################################

# Import CSV file from website
#url="https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
#response = urllib2.urlopen(url)
#cr = csv.reader(response)
#rows = [row for row in cr]

# Import CSV file from local 
with open("imports-85.data") as myfile:
    text = myfile.read().split('\n')
rows = []
for row in text[:-1]:    
    rows.append(row.split(','))
df = pd.DataFrame(data=rows)
print df.head(5)


###############################################################################
#                           Preprocessing dataframe
###############################################################################

df.columns = ["symbolizing","normalized-losses","make","fuel-type","aspiration",
              "num-of-doors","body-style","drive-wheels","engine-location",
              "wheel-base","length","width","height","curb-weight","engine-type",
              "num-of-cylinders","engine-size","fuel-system","bore","stroke",
              "compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg",
              "price"]

df=df.replace(to_replace ="?",value =np.nan)
df=df.astype({"symbolizing":int,"normalized-losses":float,"wheel-base":float,
              "length":float,"width":float,"height":float,"curb-weight":int,
              "engine-size":int,"bore":float,"stroke":float,
              "compression-ratio":float,"horsepower":float,"peak-rpm":float,
              "city-mpg":int,"highway-mpg":int,"price":float},errors='ignore')

# Replace missing values
df = df.dropna(subset=["price"],axis=0)  # drop rows whos value is nan
nl_mean = df["normalized-losses"].mean()
df["normalized-losses"]=df["normalized-losses"].replace(to_replace=np.nan,value=nl_mean) 
nl_mean2 = df["horsepower"].mean()
df["horsepower"]=df["horsepower"].replace(to_replace=np.nan,value=nl_mean2) 


###############################################################################
#                           Exploratory data analysis
###############################################################################

# Z-score of "length" column
df["length"]=(df["length"]-df["length"].mean())/df["length"].std()

# Binning
bins = np.linspace(df["price"].min(),df["price"].max(),4)
group_names = ["low","medium","high"]
df["price-binned"] = pd.cut(df["price"],bins,labels=group_names,include_lowest=True)
### FIGURE 1
hist=df.hist(column="price",bins=3)

# Create dummy variables i.e. create categories for each fuel-type value and assign 0/1
df = pd.concat([df,pd.get_dummies(df["fuel-type"])],axis=1)

# Simple numerical analysis
#print df.describe(include='all')
print('')
drive_wheels_count=df["drive-wheels"].value_counts()
print 'Drive-wheels count per type:'
print drive_wheels_count
print('')

# Box plot: spot the outliers, see the distribution and skewness of the data
### FIGURE 2
boxplot2 = df.boxplot(column=["price"], by=["drive-wheels"])

# Scatter plot: spot correlation between several variables
### FIGURE 3
df.plot.scatter(x="engine-size",y="price",grid=True,title='Price as function of engine size')
df_grp=df[["drive-wheels","body-style","price"]].groupby(["drive-wheels","body-style"],as_index=False).mean()
print 'Group by drive_wheels type and body style:'
print df_grp
print('')

# Heatmap: illustrate correlation between categorical data
df_grp_pivot = df_grp.pivot(index='drive-wheels',columns='body-style',values='price')
### FIGURE 4
fig, ax = plt.subplots()
im = ax.imshow(df_grp_pivot,cmap='RdBu')
ax.set_xticks(np.arange(len(df_grp_pivot.columns)))
ax.set_yticks(np.arange(len(df_grp_pivot.index)))
ax.set_xticklabels(df_grp_pivot.columns)
ax.set_yticklabels(df_grp_pivot.index)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
ax.figure.colorbar(im, ax=ax)
plt.title('Impact of drive wheels and body style on vehicle price')

# ANOVA: find the correlation between different groups of categorical variables
# returns F-value (variation between groups divided by variations within groups)
# and p-value (confidence degree)
df_heatmp = df.groupby(["make"]).mean()["price"]
### FIGURE 5
fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
df_heatmp.sort_values(ascending=True).plot.bar()
plt.setp(ax5.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.title('Mean price for each make')

# ANOVA test between "Honda" and "Subaru"
df_anova = df[["make","price"]]
grouped_anova = df_anova.groupby(["make"])
anova_results = stats.f_oneway(grouped_anova.get_group("honda")["price"],
                               grouped_anova.get_group("subaru")["price"])  
print('ANOVA results: ')
print anova_results
# F<1  p>0.5 <-> prices between Honda and Subaru are not significantly different

anova_results = stats.f_oneway(grouped_anova.get_group("honda")["price"],
                               grouped_anova.get_group("jaguar")["price"])  
print anova_results
print('')
# F>>1 p<0.5 <-> strong correlation between categorical variable make and variable price

# Correlation: measure to what extent different variables are interdependent
### FIGURE 6
df.plot.scatter(x="engine-size",y="price",grid=True)
plt.ylim([0,50000])

# Add simple trend line
z = np.polyfit(df["engine-size"], df["price"], 1)
p1 = np.poly1d(z)
plt.plot(df["engine-size"],p1(df["engine-size"]),"k-")

### FIGURE 7
fig7 = plt.figure(7)
ax7 = fig7.add_subplot(111)
df.plot.scatter(x="highway-mpg",y="price",ax=ax7,grid=True)
plt.ylim([0,50000])

# Add simple trend line
z = np.polyfit(df["highway-mpg"], df["price"], 1)
p2 = np.poly1d(z)
plt.plot(df["highway-mpg"],p2(df["highway-mpg"]),"k-")

# Pearson correlation: measure the strength of the correlation between variables
# returns c (coefficient value)
# and p-value (level of certainty)
# c ~ 1 <-> large positive correlation
# c ~ -1 <-> large negative correlation
# c ~ 0 <-> no relationship
# p ~ 0.001 strong certainty, p ~ 0.05 moderate certainty, p ~ 0.1 weak certainty
pearson_coeff, p_val = stats.pearsonr(df["horsepower"],df["price"])
print 'Pearson correlation results: '
print pearson_coeff, p_val
print('')

###############################################################################
#                           Model development
###############################################################################

# Linear regression: y = a + b*x
XX=df[["highway-mpg"]]
YY=df[["price"]]
reg = LinearRegression().fit(XX, YY)
#print reg.coef_, reg.intercept_
# obtain prediction from reg
YY_h1 = reg.predict(XX)

# Multiple linear regression: y = a + b1x1 + b2x2 + b3x3 + ...
ZZ = df[["horsepower","curb-weight","engine-size","highway-mpg"]]
reg = LinearRegression().fit(ZZ, df[["price"]])
#print reg.coef_, reg.intercept_

# Model evaluation
### FIGURE 8
fig8 = plt.figure()
ax8 = fig8.add_subplot(111)
plt.scatter(XX,YY_h1-YY)
plt.title('Residual plot of price prediction as function of highway MPG')
plt.xlabel('Highway MPG')
plt.ylabel('Residual value')

# Polynomial regression: y = a + b*x + c*x^2 + d*x^3 + ...
XX=df["highway-mpg"]
YY=df["price"]
f = np.polyfit(XX,YY,3)
p3 = np.poly1d(f)
fig7.add_subplot(111)
ax7.plot(XX, p3(XX),'k^')

# Multidimentional polynomial regression: y = a + b*x1 + c*x2 + d*x1*x2 + ...

# Constructor
Input = [('scale',StandardScaler()),('polynomial',PolynomialFeatures(degree=2)),
         ('model',LinearRegression())]  # (name of estimator, constructor)

pipe = Pipeline(Input)  # include normalization & Polynomial transform
pipe.fit(ZZ,YY)  # perform linear regression
YY_h3 = pipe.predict(ZZ)

# In-sample evaluation: evaluate how well the model perform on the current data
# MSE: mean squared error
# R^2
price_mse = mean_squared_error(df[['price']], df['price'].mean()*np.ones([len(df),1]))
engine_size_mse = mean_squared_error(df[['price']],p1(df["engine-size"]))
print 'Engine size as prediction of price MSE: ', engine_size_mse
highway_mpg_mse = mean_squared_error(df[['price']],p2(df["highway-mpg"]))
print 'Highway MPG as prediction of price MSE: ', highway_mpg_mse
three_d_mse = mean_squared_error(df[['price']],YY_h3)
print 'horsepower, curb-weight, engine-size, highway-mpg as prediction of price MSE:'
print three_d_mse

print 'Engine size as prediction of price R2: ', 1-engine_size_mse/price_mse
print 'Highway MPG as prediction of price R2: ', 1-highway_mpg_mse/price_mse
print 'horsepower, curb-weight, engine-size, highway-mpg as prediction of price R2:'
print 1-three_d_mse/price_mse

# Prediction and decision making
# Check that values make sense
# Check visualization
# Check residual plot to see if the shape of the model fits
# Check R2 value
print 'Predicted cost of a car with H-MPG=30: ',p2(30)
print('')

###############################################################################
#                           Model evaluation
###############################################################################

# Without split into test/train sets
z = np.polyfit(df["horsepower"],df["price"],1)
p3 = np.poly1d(z)
horsepower_mse = mean_squared_error(df[['price']],p1(df["engine-size"]))
print 'Horsepower as prediction of price R2: ', 1-horsepower_mse/price_mse

# Test/Training set random selection
XX=df.drop('price', axis=1)
YY=df[['price']]
x_train,x_test,y_train,y_test = train_test_split(XX,YY,test_size=0.4,random_state=0)

# Cross-validation
lre=LinearRegression()
lre.fit(x_train[['horsepower']],y_train)
print 'R^2 without cross-validation: ',lre.score(x_test[['horsepower']],y_test)
scores = cross_val_score(LinearRegression(),df[['horsepower']],YY,cv=3)
print 'R^2 mean and STD with cross-validation: ',scores.mean(), scores.std()
print('')

# Overfitting vs underfitting: test the R2 score for different polynomial orders
x_train,x_test,y_train,y_test = train_test_split(XX,YY,test_size=0.45,random_state=0)
pol_orders = [1,2,3,4,5]
R_q = []
lre=LinearRegression()

### FIGURE 9
plt.figure(9)
plt.scatter(x_train[['horsepower']],y_train,label='Training set')
plt.scatter(x_test[['horsepower']],y_test,label='Testing set')
x_range = np.arange(df[['horsepower']].min(),0.85*df[['horsepower']].max())

for nn in pol_orders:
    pr=PolynomialFeatures(degree=nn)
    x_train_pr=pr.fit_transform(x_train[['horsepower']])  # transform data into polynomial
    x_test_pr=pr.fit_transform(x_test[['horsepower']])
    lre.fit(x_train_pr,y_train)
    R_q.append(lre.score(x_test_pr,y_test))
    plt.scatter(x_range,
            lre.predict(pr.fit_transform(x_range.reshape([len(x_range),1]))),
            label='Polynomial degree '+str(nn)+')',s=8)
plt.legend()

### FIGURE 10
plt.figure(10)    
plt.plot(pol_orders, R_q)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
    
# Ridge regression
RidgeModel = Ridge(alpha=0.1)
RidgeModel.fit(x_train[['horsepower']],y_train)
YY_h4 = RidgeModel.predict(x_range.reshape([len(x_range),1]))

# Grid Search: look for the best hyperparameters
params = [{'alpha':[0.001,0.1,1,10,100,1000]}]
Grid1 = GridSearchCV(Ridge(),params,cv=4)
Grid1.fit(df[["horsepower","curb-weight","engine-size","highway-mpg"]],df[['price']])
Grid1.best_estimator_
print Grid1.cv_results_['mean_test_score']

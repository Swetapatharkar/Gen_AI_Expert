import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression


dataset=pd.read_csv(r"C:\Users\ADMIN\Data_Science_and_AI\Spyder\Classwork\Salary_Data.csv")
dataset

#X = dataset[["YearsExperience"]]
#y = dataset["Salary"]

x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

dataset.isnull().sum()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)



regressor=LinearRegression()
regressor.fit(x_train, y_train)

y_predict=regressor.predict(x_test)


comparison=pd.DataFrame({'Actual':y_test,'prediction':y_predict})

print(comparison)

plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title("Salary of Emp based on Exp")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

m_slope=regressor.coef_
print(m_slope)


c_intercept=regressor.intercept_
print(c_intercept)


y_12_exp=m_slope * 12 + c_intercept
print(y_12_exp)

bias=regressor.score(x_train, y_train)
print(bias)

variance=regressor.score(x_test, y_test)
print(variance)


#implement Statistics

dataset.mean()
dataset['Salary'].mean()
dataset['YearsExperience'].mean()


dataset.median()
dataset['Salary'].median()
dataset['YearsExperience'].median()

dataset.mode()
dataset['Salary'].mode()
dataset['YearsExperience'].mode()


dataset.var()
dataset['Salary'].var()
dataset['YearsExperience'].var()


dataset.std()
dataset['Salary'].std()
dataset['YearsExperience'].std()

from scipy.stats import variation
variation(dataset.values)


variation(dataset['Salary'])
variation(dataset['YearsExperience'])


#Correlation
dataset.corr()
dataset['Salary'].corr(dataset['YearsExperience'])


#Skewness

dataset.skew()
dataset['Salary'].skew()


#Standard Error
dataset.sem()
dataset['Salary'].sem()
dataset['YearsExperience'].var()


#Z scpre

import scipy.stats as stats
dataset.apply(stats.zscore)
stats.zscore(dataset['Salary'])

#ANOVA
y_mean=np.mean(y)
SSR=np.sum((y_predict - y_mean)**2)
print(SSR)

y=y[0:6]
SSE=np.sum((y - y_predict)**2)
print(SSE)

mean_total=np.mean(dataset.values)
SST = np.sum((dataset.values - mean_total) ** 2)
print(SST)

r_square= 1- (SSR / SST)
print(r_square)
print(bias)
print(variance)

#Save the model for further use in UI
filename='linear_regression_model.pkl'
with open(filename,'wb') as file:
    pickle.dump(regressor,file)
print("Model has been pickled and saved as Linear_regressor_model.pkl")    

import os
print(os.getcwd())

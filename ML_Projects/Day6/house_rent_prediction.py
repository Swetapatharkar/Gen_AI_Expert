#Import Rerquired Libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib


#Import Dataset
houserent=pd.read_csv(r'C:\Users\ADMIN\Data_Science_and_AI\Spyder\Self_MachineLearning\House_Rent_Prediction_India\house_rents_in_india.csv')

#Check columns
houserent.columns

#Define x and y dataset
#x=houserent[['area','area_rate','beds','bathrooms','furnishing']] -- model was not working properly so changes X dataset

#x = houserent[['area','area_rate','beds','bathrooms','furnishing','city']]  --- not working

x = houserent[['area','beds','bathrooms','furnishing','city']]
x=pd.get_dummies(x,drop_first=True)
houserent['total_rooms'] = houserent['beds'] + houserent['bathrooms']
y = np.log(houserent['rent'])

# Variable relationship analysis
houserent = houserent[houserent['rent'] < 200000]
houserent['city'].value_counts().head(10)

#Split data into Training and Testing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
print(x_train)
print(y_train)

#Feature Scaling
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# =============================================================================
# 
# model=LinearRegression()
# model.fit(x_train,y_train)  --- did not work 
# =============================================================================

#Train model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(x_train, y_train)

#Prediction
y_pred=model.predict(x_test)
print(y_pred)

#Evaluate Model
print("MAE:", mean_absolute_error(y_test,y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# We can see that 
# =============================================================================
# MAE ≈ 26,219
# On average, your prediction is off by ~₹26K
# MSE ≈ 8.9e9
# Large because errors are squared (units blow up)
# R² ≈ 0.23
# Model explains only 23% of variance ❌ (weak)
# =============================================================================

#Lets remove field Locality from X dataset and chanhe the regression model to Random forest

#New Data Prediction
new_data=pd.DataFrame({
    'area': [1200],
    'beds':[2],
    'bathrooms':[2],
    'furnishing':['Furnished'],
    'city':['Pune']  
      
    })

## Encode same way
new_data = pd.get_dummies(new_data, drop_first=True)

#Align with training column
new_data = new_data.reindex(columns=x.columns, fill_value=0)

#Scale
new_data = sc.transform(new_data)

#Predict
pred = model.predict(new_data)


#Check Prediction via graphs
sns.barplot(x='city', y='rent', data=houserent)
sns.violinplot(x='city', y='rent', data=houserent)
sns.boxplot(x='furnishing', y='rent', data=houserent)

# =============================================================================
# #Store the file for future prediction
# pickle.dump(model, open('house_rent_prediction.pkl','wb'))
# pickle.dump(sc,open('scaler.pkl','wb'))
# pickle.dump(x.columns,open('model_columns.pkl','wb'))
# =============================================================================

# Save model
joblib.dump(model, "model.pkl")

# Save scaler
joblib.dump(sc, "scaler.pkl")

# Save columns
joblib.dump(x.columns, "columns.pkl")

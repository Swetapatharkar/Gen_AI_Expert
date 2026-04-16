import pandas as pd
import matplotlib.pyplot as plt

platform=pd.read_csv(r"C:\Users\ADMIN\Data_Science_and_AI\Spyder\Self_MachineLearning\SocialMediaData Analysis\platform_statistics_2026.csv")
platform

platform.columns

x=platform.drop('avg_engagement_rate_pct',axis=1)
y=platform['avg_engagement_rate_pct']

platform.isnull().sum()


x=pd.get_dummies(x,drop_first=True)
print(x)

#Split into train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
  
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#from sklearn.linear_model import LinearRegression
#model=LinearRegression()
#model.fit(x_train,y_train)

#Implementing Random Forest Algorithm as Simple Linear Regression is giving large difference between actual and predicted values. 

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)




from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

print(mean_absolute_error(y_test, y_predict))
print(mean_squared_error(y_test, y_predict))
print(r2_score(y_test, y_predict))

new_pred = model.predict(x_test[:1])
print(new_pred)
new_pred2 = model.predict(x_test[:])
print(new_pred2)




comparison=pd.DataFrame({'Actual':y_test,'prediction':y_predict.round(0)})
print(comparison)

plt.scatter(y_test,y_predict)
plt.xlabel("Actual Engagement Rate")
plt.ylabel("Actual vs Predicted Values")
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_predict)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.xlabel("Actual Engagement Rate")
plt.ylabel("Predicted Engagement Rate")
plt.title("Actual vs Predicted Values")
plt.show()



importance = pd.Series(model.feature_importances_, index=x.columns)

top_10 = importance.sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
top_10.sort_values().plot(kind='barh')
plt.title("Top 10 Feature Importance")
plt.xlabel("Importance Score")
plt.show()



importance = pd.Series(model.feature_importances_, index=x.columns)

top_15 = importance.sort_values(ascending=False).head(15)

plt.figure(figsize=(12,8))
top_15.sort_values().plot(kind='barh')
plt.title("Top 15 Drivers of Engagement Rate")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
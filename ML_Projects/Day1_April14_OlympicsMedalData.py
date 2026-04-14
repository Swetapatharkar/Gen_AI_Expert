
import numpy as np
import pandas as pd
import seaborn as sns

teams=pd.read_csv(r"C:\Users\ADMIN\Data_Science_and_AI\Spyder\Self_MachineLearning\teams.csv")
teams

teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]
teams[["year", "athletes", "age", "prev_medals", "medals"]].corr()["medals"]


sns.lmplot(x='athletes',y="medals",data=teams,fit_reg=True,ci=None) # relation between number of athelete and medals won(strong)
sns.lmplot(x='age',y="medals",data=teams,fit_reg=True,ci=None) #relation between age and medals won(weak)

teams.plot.hist(y="medals")

teams[teams.isnull().any(axis=1)].head(20)
teams=teams.dropna()
teams.isnull().sum()
teams.shape


train=teams[teams["year"]< 2012].copy()

test= teams[teams["year"]>=2012].copy()

train.shape

test.shape

#### Accuracy Metric
####  We'll use mean squared error. This is a good default regression accuracy metric. 
##  It's the average of squared differences between the actual results and your predictions.

from sklearn.linear_model import LinearRegression

reg=LinearRegression()

predictors=["athletes","prev_medals"]

reg.fit(train[predictors],train["medals"])

predictions=reg.predict(test[predictors])
predictions.shape
test["predictions"]=predictions  #actual medals and predicted medals stay side by side in one table so we added this to table



print(test["predictions"].dtype)
test["predictions"] = pd.to_numeric(test["predictions"])
test.loc[test["predictions"] < 0, "predictions"] = 0        
test["predictions"] = test["predictions"].round()        
from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(test["medals"], test["predictions"])
error        
teams.describe()["medals"]
test["predictions"] = predictions
test[test["team"] == "USA"]
test[test["team"] == "IND"]  
errors = (test["medals"] - predictions).abs()
errors
error_by_team = errors.groupby(test["team"]).mean()
medals_by_team = test["medals"].groupby(test["team"]).mean()
error_ratio =  error_by_team / medals_by_team 

error_ratio = error_ratio[np.isfinite(error_ratio)]
error_ratio.plot.hist()
error_ratio.sort_values()

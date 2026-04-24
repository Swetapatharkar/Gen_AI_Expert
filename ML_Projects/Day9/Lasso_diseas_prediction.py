import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score



dataset=pd.read_csv(r'C:\Users\ADMIN\Data_Science_and_AI\Spyder\Self_MachineLearning\Diseas Prediction\Diseas_prediction.csv')

x=dataset.drop('target',axis=1)
y=dataset['target']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# =============================================================================
# scalar=StandardScaler()
# x_train=scalar.fit_transform(x_train)
# x_test=scalar.transform(x_test)
# 
# model=Lasso(alpha=0.5, max_iter=10000)
# model.fit(x_train,y_train)
# 
# print("Coefficients:",model.coef_)
# 
# y_pred = lasso.predict(X_test_scaled)
# 
# =============================================================================

# Step 4: Feature scaling (VERY IMPORTANT for LASSO)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# Step 5: Apply LASSO
lasso = Lasso(alpha=0.5, max_iter=10000)  # alpha = regularization strength
lasso.fit(x_train_scaled, y_train)

# Step 6: Predictions
y_pred = lasso.predict(x_test_scaled)

# Step 7: Evaluation
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# Step 8: Coefficients (IMPORTANT for understanding LASSO)
coefficients = pd.Series(lasso.coef_, index=x.columns)
print("\nFeature Coefficients:\n", coefficients)

# Step 9: Plot coefficients
coefficients.plot(kind='bar')
plt.title("LASSO Feature Importance (Zero means removed)")
plt.show()
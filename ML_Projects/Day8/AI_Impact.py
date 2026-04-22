import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Load dataset
df = pd.read_csv(r"C:\Users\ADMIN\Data_Science_and_AI\Spyder\Self_MachineLearning\AI_impact_on_Job_market\ai_job_impact.csv")

# 🔥 Use only required columns (IMPORTANT)
df = df[[
    'Years_Experience',
    'Education_Level',
    'Industry',
    'Automation_Risk',
    'Job_Status',          # classification target
    'Salary_Before_AI'     # regression target
]]

# Drop missing target rows
df = df.dropna(subset=['Job_Status', 'Salary_Before_AI'])

# Features
x = df[['Years_Experience', 'Education_Level', 'Industry', 'Automation_Risk']]
y_class = df['Job_Status']
y_reg = df['Salary_Before_AI']

# Categorical columns
cat_cols = ['Education_Level', 'Industry']

# Preprocessing
# =============================================================================
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
#     ],
#     remainder='passthrough'
# )
# 
# =============================================================================
# ---------------- Classification ----------------
x_train, x_test, y_train, y_test = train_test_split(
    x, y_class, test_size=0.2, random_state=42
)

# Step 1: define preprocessor FIRST
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'),
         ['Education_Level', 'Industry', 'Automation_Risk'])
    ],
    remainder='passthrough'
)

# Step 2: then use it inside pipeline
clf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])
clf_model.fit(x_train, y_train)

print("Classification Accuracy:", clf_model.score(x_test, y_test))

# ---------------- Regression ----------------
xr_train, xr_test, yr_train, yr_test = train_test_split(
    x, y_reg, test_size=0.2, random_state=42
)

reg_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

reg_model.fit(xr_train, yr_train)

print("Regression Score:", reg_model.score(xr_test, yr_test))

# Save models
joblib.dump(clf_model, "clf_model.pkl")
joblib.dump(reg_model, "reg_model.pkl")

print("✅ Models saved!")
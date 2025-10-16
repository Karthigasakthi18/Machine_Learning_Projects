# Task 2: Multiple Regression Model for Store Sales Forecasting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# Step 1: Load datasets
sales = pd.read_csv('train.csv')
stores = pd.read_csv('stores.csv')
holidays = pd.read_csv('holidays_events.csv')

print("✅ Data loaded successfully")

# Step 2: Convert date to datetime
sales['date'] = pd.to_datetime(sales['date'])

# Step 3: Feature Engineering
# Merge store info
data = sales.merge(stores, on='store_nbr', how='left')

# Merge holiday info
holidays['date'] = pd.to_datetime(holidays['date'])
holidays['is_holiday'] = 1
holiday_flags = holidays[['date', 'is_holiday']].drop_duplicates()
data = data.merge(holiday_flags, on='date', how='left')
data['is_holiday'].fillna(0, inplace=True)

# Extract time-based features
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['dayofweek'] = data['date'].dt.dayofweek

# Step 4: Encode categorical variables (store type, cluster, family)
data = pd.get_dummies(data, columns=['type', 'family'], drop_first=True)

# Step 5: Select features
features = [
    'onpromotion', 'is_holiday', 'cluster', 'year',
    'month', 'day', 'dayofweek'
]
# Add the encoded categorical columns
features += [col for col in data.columns if col.startswith('type_') or col.startswith('family_')]

X = data[features]
y = data['sales']

# Step 6: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the Multiple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Predict on test set
y_pred = model.predict(X_test)

# Step 9: Evaluate Model
rmse = sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Step 10: Plot Actual vs Predicted
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()

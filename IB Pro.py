# Task 2: Simple Sales Forecast with Moving Average

import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv('train.csv')

# Show basic info
print("Dataset Preview:")
print(data.head())
print("\nColumns in dataset:", data.columns)

# Step 2: Convert date column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Step 3: Aggregate sales by date (sum of all stores & families)
daily_sales = data.groupby('date')['sales'].sum().reset_index()

# Step 4: Compute Weekly and Monthly Moving Averages
daily_sales['Weekly_MA'] = daily_sales['sales'].rolling(window=7).mean()  # 7-day window
daily_sales['Monthly_MA'] = daily_sales['sales'].rolling(window=30).mean()  # 30-day window

# Step 5: Compare Forecast vs Actual
# We'll assume the latest moving average as our forecast
last_week_forecast = daily_sales['Weekly_MA'].iloc[-1]
last_month_forecast = daily_sales['Monthly_MA'].iloc[-1]

print("\n🔹 Forecast Results:")
print(f"Next week's forecast (based on 7-day average): {last_week_forecast:.2f}")
print(f"Next month's forecast (based on 30-day average): {last_month_forecast:.2f}")

# Step 6: Plot trend lines
plt.figure(figsize=(12, 6))
plt.plot(daily_sales['date'], daily_sales['sales'], label='Actual Sales', color='blue', alpha=0.6)
plt.plot(daily_sales['date'], daily_sales['Weekly_MA'], label='7-day Moving Average', color='orange')
plt.plot(daily_sales['date'], daily_sales['Monthly_MA'], label='30-day Moving Average', color='red')
plt.title('Sales Forecast with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 7: Save results to a new CSV (optional)
daily_sales.to_csv('sales_with_moving_avg.csv', index=False)
print("\n✅ Saved file: sales_with_moving_avg.csv")

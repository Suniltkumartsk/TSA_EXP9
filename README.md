# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 01-10-25

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import random

# Load dataset
data = pd.read_csv("usedcarssold.csv")

# Add random dates (if not present)
if 'Date' not in data.columns:
    data['Date'] = data['Year'].apply(lambda y: pd.Timestamp(year=int(y),
                                                             month=random.randint(1, 12),
                                                             day=random.randint(1, 28)))

# Set Date as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data.sort_index()

# Select numeric columns and resample monthly
numeric_data = data.select_dtypes(include=np.number)
data = numeric_data.resample('M').mean()

# Fill missing values (important fix)
data = data.fillna(method='ffill').fillna(method='bfill')

# ARIMA function
def arima_model(data, target_variable, order):
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]
    
    # Fit ARIMA model
    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()
    
    # Forecast
    forecast = fitted_model.forecast(steps=len(test_data))
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data', color='blue')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data', color='green')
    plt.plot(test_data.index, forecast, label='Forecasted Data', color='red', linestyle='--')
    plt.title(f'ARIMA Forecasting for {target_variable}', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(target_variable, fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Root Mean Squared Error (RMSE): {round(rmse, 4)}")

# Run ARIMA for Sold_Cars
arima_model(data, 'Sold_Cars', order=(5, 1, 0))
```

### OUTPUT:
<img width="995" height="634" alt="image" src="https://github.com/user-attachments/assets/8fbe4782-af8e-4ac7-bb7e-532989e92f50" />


### RESULT:
Thus the program run successfully based on the ARIMA model using python.

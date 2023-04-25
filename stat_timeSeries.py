# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 09:03:27 2023

@author: Gowtham S
"""




import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

df = pd.read_csv("u2k_i200.csv") # changing mock data to our customer data


columns = ['partner_transaction_id', 'member_id', 'optimized_date',
       'parent_merchant_format_name', 'merchant_format_name',
       'transaction_amount', 'member_home_state', 'category', 'subcategory']
X = df[columns]

size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

# ARIMA model
# walk-forward validation
for t in range(len(test)):
 model = ARIMA(history, order=(5,1,0)) # p, d, q values has to change
 model_fit = model.fit()
 output = model_fit.forecast()
 yhat = output[0]
 predictions.append(yhat)
 obs = test[t]
 history.append(obs)
 print('predicted=%f, expected=%f' % (yhat, obs))

# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

################################################################################

# ARIMA + Random Forest Regressor

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split

columns = ['partner_transaction_id', 'member_id', 'optimized_date',
       'parent_merchant_format_name', 'merchant_format_name',
       'member_home_state', 'category', 'subcategory']

# Changing data and have to try this clearly for ARIMA as well
X = df[columns]
y = df['transaction_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = False)

# Choose the ARIMA model parameters (p, d, q)
p = 2
d = 1
q = 2

# Fit the ARIMA model
model = ARIMA(train, order=(p, d, q))
model_fit = model.fit()

# Print model summary
print(model_fit.summary())

# Calculate the residuals
residuals = y_train - model_fit.fittedvalues

from sklearn.ensemble import RandomForestRegressor


# Train the Random Forest model
model_rf = RandomForestRegressor(n_estimators=150)
model_rf.fit(X_train, residuals) # Not scaling data

# Make predictions with the ARIMA model
arima_predictions = model_fit.forecast(steps=len(X_test))

# Make predictions with the Random Forest model
rf_predictions = model_rf.predict(X_test)

# Combine the ARIMA and Random Forest predictions
final_predictions = arima_predictions + rf_predictions

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Calculate R2
r2 = r2_score(y_test, final_predictions)

# Calculate MAE
mae = mean_absolute_error(y_test, final_predictions)

# Calculate MSE
mse = mean_squared_error(y_test, final_predictions)

# Print the results
print(f'R2: {r2:.2f}')
print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')



"""
ERROR: Got some error based on data have to pre-process

04/09 - meeting with Mao sir told to use data_utils and preprocess under TGAT
will add below
"""







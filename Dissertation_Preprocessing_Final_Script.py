#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the Excel file.
file_path = '/Users/burcince/Desktop/Dissertation Data Sheets/Dissertation_Financial_Data.xlsx'
sheet_name = 'Sheet1'
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Ensure Date column is datetime type
df['Date'] = pd.to_datetime(df['Date'])

# Set Date as the index
df.set_index('Date', inplace=True)

# Forward fill missing PX_Last data
df['PX_Last'].fillna(method='ffill', inplace=True)

# Calculate Moving Averages
df['MA_7'] = df.groupby('Equity')['PX_Last'].transform(lambda x: x.rolling(window=7).mean())
df['MA_28'] = df.groupby('Equity')['PX_Last'].transform(lambda x: x.rolling(window=28).mean())


# In[2]:


def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = df.groupby('Equity')['PX_Last'].transform(lambda x: calculate_rsi(x, window=14))


# In[3]:


# Calculation of short (3 weeks) / mid (12 weeks) / long (1 year) returns of Stock Prices [PX_Last], by comparing
# stock price differences between a given state and the defined time periods after.

def calculate_returns(prices, periods):
    returns = prices.pct_change(periods=periods)
    return returns

# Calculate short/mid/long term returns (3 weeks, 12 weeks, 52 weeks)
df['Short_Term_Return'] = -df.groupby('Equity')['PX_Last'].transform(lambda x: calculate_returns(x, periods=-3))
df['Mid_Term_Return'] = -df.groupby('Equity')['PX_Last'].transform(lambda x: calculate_returns(x, periods=-12))
df['Long_Term_Return'] = -df.groupby('Equity')['PX_Last'].transform(lambda x: calculate_returns(x, periods=-52))




# In[4]:


# Categorizing the returns with respect to their percentages.

def categorize_return(return_value):
    if return_value <= -0.05:
        return -2
    elif return_value > -0.05 and return_value <= -0.01:
        return -1
    elif return_value > -0.01 and return_value < 0.01:
        return 0
    elif return_value >= 0.01 and return_value < 0.05:
        return 1
    elif return_value >= 0.05:
        return 2

df['Short_Term_Return_Category'] = df['Short_Term_Return'].apply(categorize_return)
df['Mid_Term_Return_Category'] = df['Mid_Term_Return'].apply(categorize_return)
df['Long_Term_Return_Category'] = df['Long_Term_Return'].apply(categorize_return)


# In[5]:


# Creating binary variables for short/mid/long term returns, regarding whether they are positive or not.
def binary_category(return_sign):
    if return_sign>=1:
        return 1
    elif return_sign<1:
        return 0

df['Short_Term_Positive_Binary'] = df['Short_Term_Return_Category'].apply(binary_category)
df['Mid_Term_Positive_Binary'] = df['Mid_Term_Return_Category'].apply(binary_category)
df['Long_Term_Positive_Binary'] = df['Long_Term_Return_Category'].apply(binary_category)


# In[6]:


# Save the results to a new Excel file.
df.to_excel('/Users/burcince/Desktop/Dissertation Data Sheets/Data_W_MissingFields.xlsx')

# Display the first few rows to verify.
print(df.head())


# In[7]:


df


# In[10]:


# Script for handling missing data, the data missing due to the nature of the time series calculations on Moving
# Average, RSI & Returns

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the preprocessed data, which has the missing data
file_path = '/Users/burcince/Desktop/Dissertation Data Sheets/Data_W_MissingFields.xlsx'
df = pd.read_excel(file_path)

# Ensure 'Date' column is datetime type if it's in the data
df['Date'] = pd.to_datetime(df['Date'])

# Create a unique index combining 'Equity' and 'Date'
df.set_index(['Equity', 'Date'], inplace=True)

# Function to fill missing values using linear regression
def fill_missing_with_regression(group, col_name):
    # Extract indices and values for non-missing entries
    non_missing = group[~group[col_name].isnull()]
    missing = group[group[col_name].isnull()]
    
    if non_missing.empty or missing.empty:
        return group

    X_train = non_missing['Week_Index'].values.reshape(-1, 1)
    y_train = non_missing[col_name].values

    # Fit the regression model
    model = LinearRegression().fit(X_train, y_train)
    
    # Predict missing values
    X_missing = missing['Week_Index'].values.reshape(-1, 1)
    group.loc[group[col_name].isnull(), col_name] = model.predict(X_missing)
    
    return group

# Apply the function to each equity group, for the columns that need backward filling and where the amount of weeks
# is neglitible that linear regression is applicable
df = df.groupby('Equity').apply(lambda group: fill_missing_with_regression(group, 'MA_7'))
df = df.groupby('Equity').apply(lambda group: fill_missing_with_regression(group, 'RSI'))


# Apply the function to each equity group, for the columns that need forward filling and where the amount of weeks
# is neglitible that linear regression is applicable
df = df.groupby('Equity').apply(lambda group: fill_missing_with_regression(group, 'Short_Term_Return'))

# For longer time periods where there are more weeks of missing data, fit a polynomial regression. Repeat the same process

# Function to fill missing values using polynomial regression
def fill_missing_with_polynomial_regression(group, col_name, degree=3):
    # Extract indices and values for non-missing entries
    non_missing = group[~group[col_name].isnull()]
    missing = group[group[col_name].isnull()]
    
    if non_missing.empty or missing.empty:
        return group

    X_train = non_missing['Week_Index'].values.reshape(-1, 1)
    y_train = non_missing[col_name].values

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)

    # Fit the polynomial regression model
    model = LinearRegression().fit(X_train_poly, y_train)
    
    # Predict missing values
    X_missing = missing['Week_Index'].values.reshape(-1, 1)
    X_missing_poly = poly.transform(X_missing)
    group.loc[group[col_name].isnull(), col_name] = model.predict(X_missing_poly)
    
    return group

# Apply the function to each equity group for mid term / long term returns and for Moving Avg. 28 Weeks
return_columns = ['Mid_Term_Return', 'Long_Term_Return','MA_28']

for col in return_columns:
    df = df.groupby('Equity').apply(lambda group: fill_missing_with_polynomial_regression(group, col, degree=3))

return_columns = ['PE_RATIO', 'Price2CashF']

for col in return_columns:
    df = df.groupby('Equity').apply(lambda group: fill_missing_with_polynomial_regression(group, col, degree=7))

# As the missing values are now completed for returns, the category labelling and binary one-hot encoding needs
# to be repeated

# Categorizing the returns with respect to their percentages.

def categorize_return(return_value):
    if return_value <= -0.05:
        return -2
    elif return_value > -0.05 and return_value <= -0.01:
        return -1
    elif return_value > -0.01 and return_value < 0.01:
        return 0
    elif return_value >= 0.01 and return_value < 0.05:
        return 1
    elif return_value >= 0.05:
        return 2

df['Short_Term_Return_Category'] = df['Short_Term_Return'].apply(categorize_return)
df['Mid_Term_Return_Category'] = df['Mid_Term_Return'].apply(categorize_return)
df['Long_Term_Return_Category'] = df['Long_Term_Return'].apply(categorize_return)


# Creating binary variables for short/mid/long term returns, regarding whether they are positive or not.
def binary_category(return_sign):
    if return_sign>=1:
        return 1
    elif return_sign<1:
        return 0

df['Short_Term_Positive_Binary'] = df['Short_Term_Return_Category'].apply(binary_category)
df['Mid_Term_Positive_Binary'] = df['Mid_Term_Return_Category'].apply(binary_category)
df['Long_Term_Positive_Binary'] = df['Long_Term_Return_Category'].apply(binary_category)



# In[11]:


# Save the results to a new Excel file.

df.to_excel('/Users/burcince/Desktop/Dissertation Data Sheets/Preprocessed_Financial_Data.xlsx')

# Display the first few rows to verify.
print(df.head())


# In[12]:


df


# In[24]:


import pandas as pd
import numpy as np
from scipy.stats import norm

# Load the combined data
file_path = '/Users/burcince/Desktop/Dissertation Data Sheets/Data_With_Market_Metrics.xlsx'
df = pd.read_excel(file_path)

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Calculate Market Returns
df['Market_Return'] = df['Market_Index'].pct_change()

# Define rolling window size (e.g., 52 weeks for yearly rolling metrics)
window_size = 52

# Calculate Rolling Volatility
df['Rolling_Volatility'] = df.groupby('Equity')['RETURN'].transform(lambda x: x.rolling(window_size).std())

# Calculate Rolling Beta
def calculate_rolling_beta(group, window_size):
    rolling_cov = group['RETURN'].rolling(window_size).cov(group['Market_Return'])
    rolling_var = group['Market_Return'].rolling(window_size).var()
    return rolling_cov / rolling_var

df['Rolling_Beta'] = df.groupby('Equity').apply(lambda x: calculate_rolling_beta(x, window_size)).reset_index(level=0, drop=True)

# Calculate Rolling Value at Risk (VaR) at 95% confidence level
confidence_level = 0.95
z_score = norm.ppf(confidence_level)
df['Rolling_VaR'] = df.groupby('Equity')['RETURN'].transform(lambda x: x.rolling(window_size).quantile(1 - confidence_level))

# Calculate Rolling Sharpe Ratio
risk_free_rate = df['ShortTerm_Int'].mean() / 52  # Convert annual risk-free rate to weekly

def calculate_rolling_sharpe_ratio(group, window_size, risk_free_rate):
    excess_return = group['RETURN'] - risk_free_rate
    rolling_excess_return = excess_return.rolling(window_size).mean()
    rolling_volatility = group['RETURN'].rolling(window_size).std()
    return rolling_excess_return / rolling_volatility

df['Rolling_Sharpe_Ratio'] = df.groupby('Equity').apply(lambda x: calculate_rolling_sharpe_ratio(x, window_size, risk_free_rate)).reset_index(level=0, drop=True)

# Save the data with dynamic risk measures to a new Excel file
df.to_excel('/Users/burcince/Desktop/Dissertation Data Sheets/Data_With_Risk_Measures.xlsx', index=False)


# In[34]:


import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load the data

df = pd.read_excel('/Users/burcince/Desktop/Dissertation Data Sheets/Data_With_Risk_Measures.xlsx')

# Convert 'Date' column to datetime format if applicable
df['Date'] = pd.to_datetime(df['Date'])

# Create a unique index combining 'Equity' and 'Date'
df.set_index(['Equity', 'Date'], inplace=True)

# Function to fill missing values using polynomial regression
def fill_missing_with_polynomial_regression(group, col_name, degree=2):
    # Extract indices and values for non-missing entries
    non_missing = group[~group[col_name].isnull()]
    missing = group[group[col_name].isnull()]
    
    if non_missing.empty or missing.empty:
        return group

    X_train = non_missing['Week_Index'].values.reshape(-1, 1)
    y_train = non_missing[col_name].values

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)

    # Fit the polynomial regression model
    model = LinearRegression().fit(X_train_poly, y_train)
    
    # Predict missing values
    X_missing = missing['Week_Index'].values.reshape(-1, 1)
    X_missing_poly = poly.transform(X_missing)
    group.loc[group[col_name].isnull(), col_name] = model.predict(X_missing_poly)
    
    return group

# Specify the columns that need missing values filled
columns_to_fill = ['Rolling_Volatility', 'Rolling_Beta', 'Rolling_VaR','Rolling_Sharpe_Ratio']  

# Apply the function to each equity group for the specified columns
for col in columns_to_fill:
    df = df.groupby('Equity').apply(lambda group: fill_missing_with_polynomial_regression(group, col, degree=2))
    

# Save the final data frame which is ready for analysis
df.to_excel('/Users/burcince/Desktop/Dissertation Data Sheets/Final_Data_Frame.xlsx', index=False)


# In[33]:





# In[ ]:







# In[ ]:





# In[ ]:





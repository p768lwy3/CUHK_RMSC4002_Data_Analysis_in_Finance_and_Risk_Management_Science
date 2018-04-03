import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import seaborn
seaborn.set()

# 2.1.1 Read the raw data with pandas.read_csv()
df = pd.read_csv('imports-85.data',
            header=None,
            names=['Symboling', 'Losses', 'Make', 'Fuel_Type', 'Aspiration', 'Num_of_Doors',
                   'Body_Style', 'Drive_Wheels', 'Engine_Location', 'Wheel_Base', 'Length',
                   'Width', 'Height', 'Curb_Weight', 'Engine_Type', 'Num_of_Cylinders',
                   'Engine_Size', 'Fuel_System', 'Bore', 'Stroke', 'Compression_Ratio',
                   'Horsepower', 'Peak_rpm', 'City_mpg', 'Highway_mpg', 'Price'],
            na_values=('?'))

# 2.1.2 Data Cleaning: remove the data sample with missing values
# Add scripts to lr_rawdata.py to remove all samples/rows that have NaN values.
df.dropna(axis=0, how='any', inplace=True)

# 2.1.3 Data standardization
# Add scripts to lr_rawdata.py to split the dataframe df into training data and test data.
split_idx = int(df.shape[0] * .2)
df_train, df_test = df[split_idx:], df[:split_idx]
# Add scripts to lr_rawdata.py to standardize both the training data and test data of engine
# size and price using StandardScaler as follows:
from sklearn.preprocessing import StandardScaler
engine_size_stdscaler = StandardScaler()
df_train.loc[:, 'Std_Engine_Size'] = engine_size_stdscaler.fit_transform(df_train.Engine_Size.values.reshape(-1, 1))
df_test.loc[:, 'Std_Engine_Size'] = engine_size_stdscaler.transform(df_test.Engine_Size.values.reshape(-1, 1))
price_stdscaler = StandardScaler()
df_train.loc[:, 'Std_Price']= price_stdscaler.fit_transform(df_train.Price.values.reshape(-1, 1))
df_test.loc[:, 'Std_Price'] = price_stdscaler.transform(df_test.Price.values.reshape(-1, 1))

# 2.1.4 Linear regression on the preprocessed data
# Add scripts to lr_rawdata.py to build a linear regression model of standardized price as 
# the label on standardized horsepower (as the only feature) using the standardized training
# data.
horsepower_stdscaler = StandardScaler()
df_train.loc[:, 'Std_Horsepower'] = horsepower_stdscaler.fit_transform(df_train.Horsepower.values.reshape(-1, 1))
df_test.loc[:, 'Std_Horsepower'] = horsepower_stdscaler.transform(df_test.Horsepower.values.reshape(-1, 1))
X_train, X_test, y_train, y_test = df_train['Std_Horsepower'], df_test['Std_Horsepower'], df_train['Std_Price'], df_test['Std_Price']
linreg = linear_model.LinearRegression().fit(X_train.values.reshape(-1, 1), y_train)
"""linreg.score(X_train.values.reshape(-1, 1), y_train)"""
y_test_pred = linreg.predict(X_test.values.reshape(-1, 1))
# Add scripts to lr_rawdata.py to plot the standardized price test data versus standardized horsepower test 
# data, and the price predictions on the standardized horsepower test data versus standardized horsepower 
# test data in the same plot using different markers and colors.
true = plt.scatter(X_test, y_test, c='k', marker='.')
pred = plt.scatter(X_test, y_test_pred, c='b', marker='x')
plt.legend([true, pred], ['True', 'Prediction'])
plt.xlabel('Standardized horsepower')
plt.ylabel('Standardized price')
plt.title('Linear regression on clean and standardized test data')
plt.show()

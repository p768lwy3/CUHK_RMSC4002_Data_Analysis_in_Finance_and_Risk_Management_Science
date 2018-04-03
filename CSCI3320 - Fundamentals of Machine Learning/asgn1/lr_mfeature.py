import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import seaborn
seaborn.set()

df = pd.read_csv('imports-85.data',
            header=None,
            names=['Symboling', 'Losses', 'Make', 'Fuel_Type', 'Aspiration', 'Num_of_Doors',
                   'Body_Style', 'Drive_Wheels', 'Engine_Location', 'Wheel_Base', 'Length',
                   'Width', 'Height', 'Curb_Weight', 'Engine_Type', 'Num_of_Cylinders',
                   'Engine_Size', 'Fuel_System', 'Bore', 'Stroke', 'Compression_Ratio',
                   'Horsepower', 'Peak_rpm', 'City_mpg', 'Highway_mpg', 'Price'],
            na_values=('?'))
df.dropna(axis=0, how='any', inplace=True)

Xm_train, ym_train = df[['Engine_Size', 'Horsepower', 'Peak_rpm']], df['Price']

from sklearn.preprocessing import StandardScaler
engine_size_stdscaler = StandardScaler()
Xm_train.loc[:, 'Std_Engine_Size'] = engine_size_stdscaler.fit_transform(
    Xm_train.Engine_Size.values.reshape(-1, 1))
horsepower_stdscaler = StandardScaler()
Xm_train.loc[:, 'Std_Horsepower'] = horsepower_stdscaler.fit_transform(
    Xm_train.Horsepower.values.reshape(-1, 1))
peak_rpm_stdscaler = StandardScaler()
Xm_train.loc[:, 'Std_Peak_rpm'] = peak_rpm_stdscaler.fit_transform(
    Xm_train.Peak_rpm.values.reshape(-1, 1))
price_stdscaler = StandardScaler()
ym_train = price_stdscaler.fit_transform(ym_train.values.reshape(-1, 1))

# Solve multiple linear regression with normal equation.
X_matrix = Xm_train[['Std_Horsepower',
                     'Std_Engine_Size', 'Std_Peak_rpm']].values
bias_vector = np.ones((X_matrix.shape[0], 1))
X_unit = np.append(bias_vector, X_matrix, axis=1)
X_unit_t = np.transpose(X_unit)
theta = np.dot(np.dot(np.linalg.inv(
    np.dot(X_unit_t, X_unit)), X_unit_t), ym_train)

# Add scripts to lr_mfeature.py to print out the calculated theta with the following format:
print('Parameter theta calculate by normal equation: (%.4f, %.4f, %.4f, %.4f)' %
      (theta[0][0], theta[1][0], theta[2][0], theta[3][0]))

# Solve multiple linear regression with gradient descent.
# Without changing the parameters:
sgd_reg = linear_model.SGDRegressor(loss='squared_loss').fit(
    X_matrix, ym_train.reshape(ym_train.shape[0],))
# Add scripts to lr_mfeature.py to print out the calculated theta with the following format:
print('Parameter theta calculated by SGD: (%.4f, %.4f, %.4f, %.4f)' %
      (sgd_reg.intercept_[0], sgd_reg.coef_[0], sgd_reg.coef_[1], sgd_reg.coef_[2]))
# With changing the parmaters:
sgd_reg_tuned = linear_model.SGDRegressor(loss='squared_loss', penalty='none', max_iter=1e6).fit(
    X_matrix, ym_train.reshape(ym_train.shape[0],))
print('Parameter theta calculated by tuned SGD: (%.4f, %.4f, %.4f, %.4f)' %
      (sgd_reg_tuned.intercept_[0], sgd_reg_tuned.coef_[0], sgd_reg_tuned.coef_[1], sgd_reg_tuned.coef_[2]))
# Benchmark model by sklearn:
lin_reg_bm = linear_model.LinearRegression().fit(X_matrix, ym_train)
print('Parameter theta calculate by Sklearn: (%.4f, %.4f, %.4f, %.4f)' %
      (lin_reg_bm.intercept_[0], lin_reg_bm.coef_[0][0], lin_reg_bm.coef_[0][1], lin_reg_bm.coef_[0][2]))

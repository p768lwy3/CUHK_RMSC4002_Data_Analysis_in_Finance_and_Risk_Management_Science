import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

X_train = [[5.3], [7.2], [10.5], [14.7], [18], [20]]
y_train = [[7.5], [9.1], [13.2], [17.5], [19.3], [19.5]]

X_test = [[6], [8], [11], [22]]
y_test = [[8.3], [12.5], [15.4], [19.6]]

poly = PolynomialFeatures(degree=5)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# some code here
poly_reg = LinearRegression().fit(X_train_poly, y_train)
# Fill in the blank of the regression equation below
print('y1 = %.4f + %.4f x + %.4f x*x + %.4f x*x*x + %.4f x*x*x*x + %.4f x*x*x*x*x' %
      (poly_reg.intercept_[0], poly_reg.coef_[0][0], poly_reg.coef_[0][1], poly_reg.coef_[0][2], poly_reg.coef_[0][3], poly_reg.coef_[0][4]))
# Add scripts to poly_regular.py to print the score of the linear regression model 
# on the test data in the following format:
print('Linear regression (order 5) score is: ', poly_reg.score(X_test_poly, y_test))

xx = np.linspace(0, 26, 100)
xx_poly = poly.transform(xx.reshape(xx.shape[0], 1))
yy_poly = poly_reg.predict(xx_poly)

# some code here
# Add script to poly_regular.py to plot the predicted output yy_poly versus xx, 
# and also the test data (y_test versus X_test) in the same plot.
plt.plot(xx, yy_poly, c='b')
plt.scatter(X_test, y_test, c='r', marker='x')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear regression (order 5) result')
plt.show()

ridge_model = Ridge(alpha=4, normalize=False)
ridge_model.fit(X_train_poly, y_train)

# some code here
# Fill in the blank of the regression equation below
print('y1 = %.4f + %.4f x + %.4f x*x + %.4f x*x*x + %.4f x*x*x*x + %.4f x*x*x*x*x' %
      (ridge_model.intercept_[0], ridge_model.coef_[0][0], ridge_model.coef_[0][1], ridge_model.coef_[0][2], ridge_model.coef_[0][3], ridge_model.coef_[0][4]))
# Add scripts to poly_regular.py to print the score of the linear regression model 
# on the test data in the following format:
print('Ridge regression (order 5) score is: ', ridge_model.score(X_test_poly, y_test))
yy_ridge = ridge_model.predict(xx_poly)
# Add script to poly_regular.py to plot the predicted output yy_poly versus xx, 
# and also the test data (y_test versus X_test) in the same plot.
plt.plot(xx, yy_ridge, c='b')
plt.scatter(X_test, y_test, c='r', marker='x')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Ridge regression (order 5) result')
plt.show()

# Benchmark (simple linear regression):
benchmark = LinearRegression().fit(X_train, y_train)
print('Linear regression score is: ', benchmark.score(X_test, y_test))

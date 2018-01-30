import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import seaborn; seaborn.set()

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Q1. Get n_features and n_samples. (5%)
print('Number of features in the Diabetes dataset is: %s' % str(diabetes.data.shape[1]))
print('Number of samples in the Diabetes dataset is: %s' % str(diabetes.data.shape[0]))

# Q2. Find out how each feature fits the disease progression. (15%)
# which feature
# i_feature = 0

# Get the feature name
feature_names = ['Age', 'Sex', 'Body mass index', 'Average blood pressure', 'S1',
                 'S2', 'S3', 'S4', 'S5', 'S6']

# set two lists to save the results
order_list_of_feature_name = []
order_list_of_model_score = []

# loop through the name list
for i_feature, feature_name in enumerate(feature_names):

    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis, i_feature]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    # Create linear regression object
    model = linear_model.LinearRegression()

    # Train the model using the training sets
    model.fit(diabetes_X_train, diabetes_y_train)

    # Explained variance score: score=1 is perfect prediction
    model_score = model.score(diabetes_X_test, diabetes_y_test)

    # save to the lists
    order_list_of_feature_name.append(feature_name)
    order_list_of_model_score.append(model_score)

# print out the result
print('Order list of feature name is: %s' % str(order_list_of_feature_name))
print('Order list of model score is: %s' % str(order_list_of_model_score))

# Q3. Calculate the loss function. (5%)
# define mse function to compute
def mse(X_test, y_test, model):
    return np.mean((model.predict(X_test) - y_test)**2)

# read the best fit model again, which is the largest R-squared score.
diabetes_X = diabetes.data[:, np.newaxis, 2]
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]
model = linear_model.LinearRegression().fit(diabetes_X_train, diabetes_y_train)
# compute the mse
model_mse = mse(diabetes_X_test, diabetes_y_test, model)
# print the result
print('Value of the loss function for the best fitted model is: %s' % str(model_mse))

# Q4. Plot the predictions and test data. (15%)
# plot the real y
plt.scatter(diabetes_X_test, diabetes_y_test, c='r', marker='o', 
           label='real test set')
# plot the pred y
plt.scatter(diabetes_X_test, model.predict(diabetes_X_test), c='b', 
           marker='X', label='real test set')
# set the labels
plt.xlabel(feature_names[2])
plt.ylabel('Disease Progression')
plt.legend(loc='best')
plt.show()

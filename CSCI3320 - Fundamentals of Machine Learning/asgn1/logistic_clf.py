import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn import linear_model, datasets
from sklearn.cross_validation import train_test_split


n_samples = 10000

centers = [(-1, -1), (1, 1)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.8,
                  centers=centers, shuffle=False, random_state=42)

y[:n_samples // 2] = 0
y[n_samples // 2:] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
log_reg = linear_model.LogisticRegression()

# some code here
log_reg.fit(X_train, y_train)
test_pred = log_reg.predict(X_test)

# Q: Does the predictions of X_test contain values other than 0 or 1?
print((sum(test_pred == 1) + sum(test_pred == 0)) == len(test_pred))

# Add scripts to logistic_clf.py to plot the data points in X_test using the function scatter() with different colors for different predicted classes.
colors = 'br'
for i, color in zip(log_reg.classes_, colors):
    idx = np.where(test_pred == i)
    plt.scatter(X_test[idx, 0], X_test[idx, 1], c=color, cmap=plt.cm.Paired)
plt.title('Classification with Logistic Regression')
plt.xlabel('X_0')
plt.ylabel('X_1')
plt.show()

# Scatter plot of the true labels
colors = 'br'
for i, color in zip(log_reg.classes_, colors):
    idx = np.where(y_test == i)
    plt.scatter(X_test[idx, 0], X_test[idx, 1], c=color, cmap=plt.cm.Paired)
plt.title('True Classes of Data')
plt.xlabel('X_0')
plt.ylabel('X_1')
plt.show()

# Q: How many wrong predictions does the LogisticRegression estimator make on the test data?
print('Number of wrong predictions is: ', sum(y_test != test_pred))
print('Score: ', log_reg.score(X_test, y_test))
print('Number of wrong predictions is: ', (1-log_reg.score(X_test, y_test)) * len(X_test))



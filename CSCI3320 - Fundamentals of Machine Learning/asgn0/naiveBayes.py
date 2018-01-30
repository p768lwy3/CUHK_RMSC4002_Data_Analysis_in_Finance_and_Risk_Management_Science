
# coding: utf-8

# In[47]:


import numpy as np, scipy.io as sio
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB


# In[38]:


path_train = 'train.mat'
path_test = 'test.mat'


# In[14]:


# 1. Load the train.mat data. (10%)
# Read the mat file by scipy.io.
# Since the return is 'dict', take the useful parts.
X_train = sio.loadmat(path_train)['Xtrain']
y_train = sio.loadmat(path_train)['ytrain']


# In[15]:


# What are n_features and n_samples?
print('Number of features in the Diabetes dataset is: %s' % str(X_train.shape[1]))
print('Number of samples in the Diabetes dataset is: %s' % str(X_train.shape[0]))


# In[67]:


# Choose a naive claffifier fron GaussianNB, MultinomialNB, BernoulliNB for the problem.
mnb = MultinomialNB(alpha=1e-10).fit(X_train, y_train.ravel())


# In[68]:


# How accurate is the classifier on the training data?
# Calculate and print out the accuracy.
model_score = mnb.score(X_train, y_train.ravel())
print('The accuracy of the classifier: %s' % str(model_score))


# In[69]:


# 3. Load the test data test.mat and get predictions for the Xtest. (20%)
X_test = sio.loadmat(path_test)['Xtest']
y_pred = mnb.predict(X_test)
print('Number of samples in the predicted dataset is: %s' % str(y_pred.shape[0]))


# In[70]:


# Write the results to the file prediction.txt.
# Each line in prediction.txt is the corresponding label for the data point.
output_path = 'prediction.txt'
np.savetxt(output_path, y_pred, fmt='%d')


# # For Benchmark

# ### BernoulliNB

# In[71]:


bnb = BernoulliNB(alpha=1e-10).fit(X_train, y_train.ravel())
bnb_score = bnb.score(X_train, y_train.ravel())
print('The accuracy of the bernoulli naive bayes classifier: %s' % str(bnb_score))


# ### GaussianNB

# In[72]:


gnb = GaussianNB().fit(X_train, y_train.ravel())
gnb_score = gnb.score(X_train, y_train.ravel())
print('The accuracy of the gaussian naive bayes classifier: %s' % str(gnb_score))


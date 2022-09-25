# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:03:24 2019

@author: prate
"""
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_svmlight_file
from xgboost import XGBClassifier
import matplotlib.pyplot as plot

# load data in LibSVM sparse data format
X_train, y_train = load_svmlight_file("a9a.txt")
y_train[y_train == -1] = 0

X_test, y_test = load_svmlight_file("a9a.t")
y_test[y_test == -1] = 0
# split data into train and test sets
seed = 6
test_size = 0.4
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

#parameter tuning for n_estimators
model = XGBClassifier()
xgtrain = xgb.DMatrix(X_train, label = y_train)
cross_val_n = xgb.cv(params=model.get_xgb_params(), dtrain=xgtrain, num_boost_round=1000, nfold=3, early_stopping_rounds= 40)

plot.plot(cross_val_n)
plot.title('"n_estimators" Parameter Tuning')
plot.ylabel('Error')
plot.xlabel('Number of Iterations for Value of n_estimators=[1,1000]')
plot.show()
best_n_estimator = cross_val_n.shape[0]
model.n_estimators=best_n_estimator
print ("Best n_estimators value: %f\n" % best_n_estimator)

#parameter tuning for learning rate
cross_val_learning_rate = []
my_learning_rate=[0.001, 0.01, 0.1, 0.2, 0.3, 0.4]
for i in my_learning_rate:
    model.learning_rate=i
    scores = cross_val_score(model, X_train, y_train, cv=3)
    cross_val_learning_rate.append(np.mean(scores))

plot.plot(cross_val_learning_rate)
plot.title('"learning_rate" Parameter Tuning')
plot.ylabel('Cross Validation Score')
plot.xlabel('Number of Iterations for Value of learning_rate=[0.001, 0.01, 0.1, 0.2, 0.3, 0.4]')
plot.show()
best_learning_rate = my_learning_rate[np.argmax(cross_val_learning_rate)]
model.learning_rate=best_learning_rate
print ("Best learning_rate value: %f\n" % best_learning_rate)

#parameter tuning for max_depth
cross_val_max_depth = []
my_max_depth=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for i in my_max_depth:
    model.max_depth=i
    scores = cross_val_score(model, X_train, y_train, cv=3)
    cross_val_max_depth.append(np.mean(scores))

plot.plot(cross_val_max_depth)
plot.title('"Max_depth" Parameter Tuning')
plot.ylabel('Cross Validation Score')
plot.xlabel('Number of Iterations for Value of max_depth=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]')
plot.show()
best_max_depth = my_max_depth[np.argmax(cross_val_max_depth)]
model.max_depth=best_max_depth
print ("Best Max depth value: %f\n" % best_max_depth)

#parameter tuning for lambda
cross_val_lambda = []
my_lambda=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
for i in my_lambda:
    model.reg_lambda=i
    scores = cross_val_score(model, X_train, y_train, cv=3)
    cross_val_lambda.append(np.mean(scores))

plot.plot(cross_val_lambda)
plot.title('"Lambda" Parameter Tuning')
plot.ylabel('Cross Validation Score')
plot.xlabel('Number of Iterations for Value of lambda=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]')
plot.show()
best_lambda = my_lambda[np.argmax(cross_val_lambda)]
model.reg_lambda=best_lambda
print ("Best lambda value: %f\n" % best_lambda)

#model = XGBClassifier(n_estimators = best_n_estimator, max_depth=best_max_depth, reg_lambda=best_lambda, learning_rate=best_learning_rate)
scores = cross_val_score(model, X_train, y_train, cv=3)
model.fit(X_train, y_train, verbose=True)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

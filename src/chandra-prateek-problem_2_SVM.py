# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:03:24 2019

@author: prate
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_svmlight_file
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import matplotlib.pyplot as plot

# load data in LibSVM sparse data format
X_train, y_train= load_svmlight_file("a9a.txt")
X_test, y_test= load_svmlight_file("a9a.t")
# split data into train and test sets
seed = 6
test_size = 0.4
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# fit model on training data

# for simplicity we fit based on default values of hyperparameters

#tuning the C parameter
cross_val_c = [] 
my_C = [1, 10, 100, 200, 300, 400, 500]
for i in my_C:
    model = svm.SVC(C=i, kernel = 'linear')
    scores = cross_val_score(model, X_train, y_train, cv=3)
    cross_val_c.append(np.mean(scores))

plot.plot(cross_val_c)
plot.title('"C" Parameter Tuning')
plot.ylabel('Cross Validation Score')
plot.xlabel('Number of Iterations for Value of C = [1, 10, 100, 200, 300, 400, 500]')
plot.show()
best_C = my_C[np.argmax(cross_val_c)]
print ("Best C value: %f\n" % best_C)

#tuning the gamma parameter
cross_val_gamma = []
my_gamma = [0.001, 0.01, 0.1, 1]
for j in my_gamma:
    model = svm.SVC(gamma=j, kernel = 'linear')
    scores = cross_val_score(model, X_train, y_train, cv=3)
    cross_val_gamma.append(np.mean(scores))

plot.plot(cross_val_gamma)
plot.title('"Gamma" Parameter Tuning')
plot.ylabel('Cross Validation Score')
plot.xlabel('Number of Iterations for Value of Gamma = [0.001, 0.01, 0.1, 1]')
plot.show()
best_gamma = my_gamma[np.argmax(cross_val_gamma)]
print ("Best Gamma value: %f\n" % best_gamma)

cross_val_c_gamma = []
for k in my_gamma:
    for l in my_C:
        print ("Gamma value: %f" % k)
        print ("C value: %f" % l)
        model = svm.SVC(gamma=k, C = l, kernel = 'linear')
        scores = cross_val_score(model, X_train, y_train, cv=3)
        cross_val_c_gamma.append(np.mean(scores))
        print ("Score: %f\n" % np.mean(scores))

plot.plot(cross_val_c_gamma)
plot.title('"Gamma" Parameter Tuning')
plot.ylabel('Cross Validation Score')
plot.xlabel('Number of Iterations for Value of Gamma = [0.001, 0.01, 0.1, 1]')
plot.show()

"""
#using the tuned parameters
model = svm.SVC(gamma=0.01, C = 35)
scores = cross_val_score(model, X_train, y_train, cv=3)
validation_score = np.mean(scores)
print ("Validation Score: %f\n" % validation_score)
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
"""

"""
#tuning the C parameter again using hold out cross validation
cross_val_c = [] 
accuracy = []
my_C = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
for i in my_C:
    print ("C: %f" % i)
    model = svm.SVC(gamma = 0.01, C=i)
    scores = cross_val_score(model, X_train, y_train, cv=3)
    cross_val_c.append(np.mean(scores))
    
    validation_score = np.mean(scores)
    print ("Validation Score: %f" % validation_score)
    model.fit(X_train, y_train)
    
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    
    # evaluate predictions
    accuracy_score1=accuracy_score(y_test, predictions)
    accuracy.append(accuracy_score1)
    print("Accuracy: %.2f%%\n" % (accuracy_score1 * 100.0))
"""
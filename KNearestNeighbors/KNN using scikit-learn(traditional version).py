# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 21:44:33 2019

@author: z
"""

import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd


df = pd.read_csv("breast-cancer-wisconsin.data.txt")
df.replace('?', -999999, inplace=True)  ###try to make '?' as an outlier 
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

accuracies = []
for i in range(25):    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)
    
    accuracy = clf.score(X_test, y_test)
    
###just a small example:    
    example = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]])
    example_reshape = example.reshape(len(example), -1)
    prediction = clf.predict(example_reshape)
    
    accuracies.append(accuracy)
    
print(sum(accuracies)/len(accuracies))
    




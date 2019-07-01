# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 20:31:05 2019

@author: z
"""
###KNN code self-leaerning(self-made KNN algorithm)
import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random

def k_nearest_neighbours(data, predict, k=5):
    if len(data) >= k:
        warnings.warn('K is set to a value less than the total number of all voting groups!')
        distances = [] ##Here we use Euclidean distances for convenience reasons
        for group in data:
            for features in data[group]:
                euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
                distances.append([euclidean_distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    
    return vote_result, confidence

accuracies = []
for i in range(25):
    df = pd.read_csv("breast-cancer-wisconsin.data.txt")
    df.replace('?', -999999, inplace=True)  ###try to make '?' as an outlier 
    df.drop(['id'], 1, inplace=True)
    full_data = df.astype(float).values.tolist()
    random.shuffle(full_data)
    
    test_size = 0.2
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size)*len(full_data):]
    
    for i in train_data:
        train_set[i[-1]].append(i[:-1])
    for i in test_data:
        test_set[i[-1]].append(i[:-1])   
    
    correct = 0
    total = 0 
    
    for group in test_set:
        for data in test_set[group]:
             vote,Confidence = k_nearest_neighbours(train_set, data, k=5)
             if group == vote:
                 correct += 1
             else:
             total += 1
            
    print('Accuracy of self-made KNN algorithm is:', correct/total)
    accuracies.append(correct/total)

print(sum(accuracies)/len(accuracies))




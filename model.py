# -*- coding: utf-8 -*-
"""
Created on Tue May  5 14:58:16 2020

@author: HP
"""



import pandas as pd
import sklearn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
   
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')


knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train.iloc[:,1:].astype(int), y_train.iloc[:,1:].astype(int))

import pickle 
pickle.dump(knn,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

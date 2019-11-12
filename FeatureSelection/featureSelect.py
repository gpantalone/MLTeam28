import pandas as pd
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing

# read csv into np array
contData = genfromtxt('contFeaturesNormed.csv', delimiter=',')
catData = genfromtxt('categoryFeatures.csv', delimiter=',', dtype=str, encoding='UTF-8')
labels = genfromtxt('normedLabels.csv', delimiter=',')

#remove first entry, is bugged for some reason
contData = np.delete(contData, 0,0)
catData = np.delete(catData, 0,0)
labels = np.delete(labels, 0,0)

#need to encode the categorical data
#gets the unique values for each column
entry0 = np.unique(catData[:,0])
entry1 = np.unique(catData[:,1])
entry2 = np.unique(catData[:,2])
entry3 = np.unique(catData[:,3])
entry4 = np.unique(catData[:,4])
entry5 = np.unique(catData[:,5])
entry6 = np.unique(catData[:,6])

#set up the encoder, replace each cat column with its respective encoding
le = preprocessing.LabelEncoder()
le.fit(entry0)
catData[:,0] = le.transform(catData[:,0])

le = preprocessing.LabelEncoder()
le.fit(entry1)
catData[:,1] = le.transform(catData[:,1])

le = preprocessing.LabelEncoder()
le.fit(entry2)
catData[:,2] = le.transform(catData[:,2])

le = preprocessing.LabelEncoder()
le.fit(entry3)
catData[:,3] = le.transform(catData[:,3])

le = preprocessing.LabelEncoder()
le.fit(entry4)
catData[:,4] = le.transform(catData[:,4])

le = preprocessing.LabelEncoder()
le.fit(entry5)
catData[:,5] = le.transform(catData[:,5])

le = preprocessing.LabelEncoder()
le.fit(entry6)
catData[:,6] = le.transform(catData[:,6])

#normalize these new values
catData = catData.astype('float64')
catData = catData / catData.max(axis=0)
print(catData)

#combine the features back together, order is budget  company country director    genre    rating     runtime    star     writer  year

data = np.append(contData[:,0].reshape(contData[:,0].shape[0],1),catData[:,[0,1,2,3,4]],1)
data = np.append(data, contData[:,1].reshape(contData[:,0].shape[0],1),1)
data = np.append(data, catData[:,[5,6]],1)
data = np.append(data, contData[:,2].reshape(contData[:,0].shape[0],1),1)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

###################################

tree = DecisionTreeRegressor(criterion='mse')

sel = SelectFromModel(DecisionTreeRegressor(criterion='mse'))
sel.fit(X_train, y_train)
print(sel.get_support())
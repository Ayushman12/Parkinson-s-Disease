#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv("parkinsons.csv")
X = dataset.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22,23]].values
y = dataset.iloc[:,17].values
 
#spliting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#appling PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 0)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
variance = pca.explained_variance_ratio_

#used three models of ML
# KNN Model
# SVM
# Random forest classifier

#fitting in KNN model
from sklearn.neighbors import KNeighborsClassifier
cl = KNeighborsClassifier(n_neighbors=8,p=2,metric='minkowski')
cl.fit(X_train,y_train)

#predicting Results
y1_pred = cl.predict(X_test)

#accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y1_pred)

#fitting into SVM model
from sklearn.svm import SVC
cl2 = SVC()
cl2.fit(X_train,y_train)

#result
y2_pred = cl2.predict(X_test)

#accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y2_pred)

#fitting into randon forest classifier
from sklearn.ensemble import RandomForestClassifier
cl3 = RandomForestClassifier(n_estimators = 16, criterion="entropy",random_state=0)
cl3.fit(X_train,y_train)

#results
y3_pred = cl3.predict(X_test)

#accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y3_pred)

#now checking wrong predictions with Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y3_pred)
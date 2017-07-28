#importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from sklearn.svm import SVC

liver = np.loadtxt('/home/sid/Desktop/Call Health/Liver Disorders/bupa.data',delimiter = ',',usecols=range(7))#importing dataset to dataframe

#making feature set X and target set Y
X = liver[:,0:6]
Y = liver[:,6]

#applying StratifiedKFold for better unbiased results
skf = StratifiedKFold(n_splits=6,random_state=1,shuffle=True)
skf.get_n_splits(X,Y)

for train_index,test_index in skf.split(X,Y):
     X_train, X_test = X[train_index], X[test_index]
     Y_train, Y_test = Y[train_index], Y[test_index]
    
clf = SVC(C=0.5,gamma=0.009)#applying SVC model with respective parameters
clf.fit(X_train,Y_train)#fitting the model on training set
y_pred = clf.predict(X_test)#predicting test set and storing in y_pred
print accuracy_score(Y_test,y_pred)
print f1_score(Y_test,y_pred)
print confusion_matrix(Y_test,y_pred)

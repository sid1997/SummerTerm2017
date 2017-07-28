#importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler,normalize

liver = np.loadtxt('/home/sid/Desktop/Call Health/Liver Disorders/bupa.data',delimiter = ',',usecols=range(7))#importing dataset to dataframe

#making feature set X and target set Y
X = liver[:,0:6]
Y = liver[:,6]
Y = Y-1
#X = MinMaxScaler().fit_transform(X)
#X = normalize(X, norm='l2')
#print Y
#applying StratifiedKFold for better unbiased results
skf = StratifiedKFold(n_splits=6,random_state=1,shuffle=True)
skf.get_n_splits(X,Y)

for train_index,test_index in skf.split(X,Y):
     X_train, X_test = X[train_index], X[test_index]
     Y_train, Y_test = Y[train_index], Y[test_index]
    
logreg = linear_model.LogisticRegression(C=1)#applying Logistic Regression model with respective parameters
logreg.fit(X_train,Y_train)#fitting the model on training set
y_pred = logreg.predict(X_test)#predicting test set and storing in y_pred 
print accuracy_score(Y_test,y_pred)
#print f1_score(Y_test,y_pred)
print confusion_matrix(Y_test,y_pred)

fpr, tpr, thresholds = roc_curve(Y_test,y_pred)
#print fpr
#print tpr
#print thresholds

print roc_auc_score(Y_test,y_pred)

titles = ['mcv','alkphos','sgpt','sgot','gammagt','drinks']#column names to assign as labels in scatter plots

#scatter plots of each feature with target variable
for i in range(0,6):
    plt.scatter(X[:,i],Y)
    plt.xlabel(titles[i])
    plt.ylabel('Selector')#target variable
    plt.title('Liver Disorders')
    plt.show()

colors=['red','blue']#colors assigned to each class

#scatter plots of each feature with other one and colors denote classes
for i in range(0,5):
    for j in range(i,6):
        if i!=j:
            plt.scatter(X[:,i],X[:,j],c=colors)
            plt.xlabel(titles[i])
            plt.ylabel(titles[j])
            plt.title('Liver Disorders')
            plt.show()

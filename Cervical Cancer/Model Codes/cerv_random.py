#importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,precision_score,recall_score
from sklearn.model_selection import KFold,StratifiedKFold

df=pd.read_csv('/home/sid/Desktop/Call Health/Cervical Cancer/risk_factors_cervical_cancer.csv')#reading dataset to a dataframe

df=df.replace('?',np.NaN) #replacing '?' with NaN(null)  
df = df.fillna(df.astype(float).mean()) #assigning mean of column values by considering only float values to NaN

#creating feature set X and 4 classes(target variables) named Y1,Y2,Y3,Y4
X = df.iloc[:,0:32]
Y1 = df.iloc[:,32]
Y2 = df.iloc[:,33]
Y3 = df.iloc[:,34]
Y4 = df.iloc[:,35]

#casting all the data values to their respective datatypes
X = X.astype(float)
Y1 = Y1.astype(int)
Y2 = Y2.astype(int)
Y3 = Y3.astype(int)
Y4 = Y4.astype(int)

#applying Stratified KFold to randomise training data for better unbiased results
skf1 = StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
skf1.get_n_splits(X,Y1)
#print kf
for train_index,test_index in skf1.split(X,Y1):
     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
     Y1_train, Y1_test = Y1.iloc[train_index], Y1.iloc[test_index]
    
clf1 = RandomForestClassifier(n_estimators=100)#applying Random Forest model with respective parameters
clf1.fit(X_train,Y1_train)#fitting model on training set 
y1_pred = clf1.predict(X_test)#making predictions on test set and storing it in y1_pred
#print y1_pred
#print accuracy_score(Y1_test,y1_pred)
#print f1_score(Y1_test,y1_pred, average='weighted')
#print precision_score(Y1_test,y1_pred,average='weighted')
#print recall_score(Y1_test,y1_pred,average='weighted')
print confusion_matrix(Y1_test,y1_pred)

skf2 = StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
skf2.get_n_splits(X,Y2)
for train_index,test_index in skf2.split(X,Y2):
     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
     Y2_train, Y2_test = Y2.iloc[train_index], Y2.iloc[test_index]
clf2 = RandomForestClassifier(n_estimators=100)
clf2.fit(X_train,Y2_train)
y2_pred = clf2.predict(X_test)
#print y2_pred
#print accuracy_score(Y2_test,y2_pred)
#print f1_score(Y2_test,y2_pred, average='weighted')
print confusion_matrix(Y2_test,y2_pred)

skf3 = StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
skf3.get_n_splits(X,Y3)
for train_index,test_index in skf3.split(X,Y3):
     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
     Y3_train, Y3_test = Y3.iloc[train_index], Y3.iloc[test_index]
clf3 = RandomForestClassifier(n_estimators=100)
clf3.fit(X_train,Y3_train)
y3_pred = clf3.predict(X_test)
#print y3_pred
#print accuracy_score(Y3_test,y3_pred)
#print f1_score(Y3_test,y3_pred, average='weighted')
print confusion_matrix(Y3_test,y3_pred)

skf4 = StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
skf4.get_n_splits(X,Y4)
for train_index,test_index in skf4.split(X,Y4):
     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
     Y4_train, Y4_test = Y4.iloc[train_index], Y4.iloc[test_index]
print Y4_test.value_counts()
clf4 = RandomForestClassifier(n_estimators=100)
clf4.fit(X_train,Y4_train)
y4_pred = clf4.predict(X_test)
print y4_pred
print Y4_test
print accuracy_score(Y4_test,y4_pred)
#print f1_score(Y4_test,y4_pred, average='weighted')
print confusion_matrix(Y4_test,y4_pred)
print precision_score(Y4_test,y4_pred)
print recall_score(Y4_test,y4_pred)


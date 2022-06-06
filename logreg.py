#Use the feature sets given below to construct a logistic regression model 'ed', 'employ', 'equip', 'callcard', 'wireless'
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
%matplotlib inline 
import matplotlib.pyplot as plt

df=pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv")
churn_df = df[['ed', 'employ', 'equip', 'callcard','wireless','age','address','longmon']]

#For 'ed'
churn_df['ed'] = churn_df['ed'].astype('int')
X = np.asanyarray(churn_df[['ed', 'employ', 'equip', 'callcard','wireless','age','address','longmon']])
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)#takes entire array,computes mean,variance and gets a standard value using (x-mean)/variance
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)#c is standard value.
yhat = LR.predict(X_test)# gives churn rate
u=confusion_matrix(y_test, yhat)
print(u)
acc=(u[0,0]+u[1,1])/40
print("The accuracy is:",acc)









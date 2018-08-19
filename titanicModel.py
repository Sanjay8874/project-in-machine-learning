import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import *
from sklearn.linear_model import LinearRegression
from sklearn import neighbors,preprocessing
from sklearn import svm
import matplotlib.pyplot as plt


#get train data and modifying it
train=pd.read_csv('train.csv')
train=train[['PassengerId','Survived','Pclass','Sex','Age','Fare','SibSp','Parch','Embarked']]
train.replace('male',0,inplace=True)
train.replace('female',1,inplace=True)
train.fillna(0,inplace=True)
train['Age'].replace('0',train.mean(),inplace=True)
train.replace('S',0,inplace=True)
train.replace('Q',1,inplace=True)
train.replace('C',2,inplace=True)

#get test data and modifying it
test=pd.read_csv('test.csv')
test=test[['PassengerId','Pclass','Sex','Age','Fare','SibSp','Parch','Embarked']]
test.replace('male',0,inplace=True)
test.replace('female',1,inplace=True)
test.fillna(0,inplace=True)
test['Age'].replace('0',test.mean(),inplace=True)
test.replace('S',0,inplace=True)
test.replace('Q',1,inplace=True)
test.replace('C',2,inplace=True)
test.fillna(-99999, inplace=True)



X=np.array(train.drop(['Survived'],1))
X=preprocessing.scale(X)
y=np.array(train['Survived'])

#feature Scaling


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


#K Nearest Neighbors
clf=neighbors.KNeighborsClassifier(n_jobs=-1)
clf.fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)
print("accuracy of K nearest neighbors",accuracy)
#pred=clf.predict(test)
#print(pred)

#Naive's Bayes Gaussian
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)
accuracy=classifier_nb.score(X_test,y_test)
print("accuracy of naive's bayes",accuracy)

#Linear Support Vector Machine
from sklearn.svm import SVC
classifier_svm = SVC(kernel = 'linear', random_state = 0)
classifier_svm.fit(X_train, y_train)
accuracy=classifier_svm.score(X_test,y_test)
print('accuracy of support vector classifier',accuracy)

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)
classifier_rf.fit(X_train, y_train)
accuracy=classifier_rf.score(X_test,y_test)
print('accuracy of random forest classifier',accuracy)






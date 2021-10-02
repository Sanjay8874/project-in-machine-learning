from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn import svm,linear_model,preprocessing,neighbors,naive_bayes,ensemble
import numpy as np 
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

iris=load_iris()
X=iris.data
y=iris.target

X=preprocessing.scale(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

'''
-----------------K Nearest Neighbors---------------
'''

knn=neighbors.KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)
accuracy=knn.score(X_test,y_test)
print('k nearest neighbors accuracy',accuracy)

#predicting 
knnpred=knn.predict(X_test)
#confusion matrix
knnconfusion=confusion_matrix(y_test,knnpred)

#Preparing another classifier for the Visualisation that trains the Classifier on Sepal characteristics
classifier_knnvis= neighbors.KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_knnvis.fit(X_train[:,[0,1]], y_train)

# Visualising the Test set results of K-NN
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier_knnvis.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('K-Nearest Neighbours Classification (Test set)')
plt.xlabel('Sepal Length(in cm)')
plt.ylabel('Sepal Width(in cm)')
L=plt.legend()
L.get_texts()[0].set_text('Setosa')
L.get_texts()[1].set_text('Versicolor')
L.get_texts()[2].set_text('Virginica')
plt.show()

'''
-----------------SVM----------------
'''
svml=svm.SVC(kernel='linear')
svml.fit(X_train,y_train)
accuracy=svml.score(X_test,y_test)
print('svm linear accuracy',accuracy)

#predicting 
svmpred=svml.predict(X_test)
#confusion matrix
svmconfusion=confusion_matrix(y_test,svmpred)

#Preparing another classifier for the Visualisation that trains the Classifier on Sepal characteristics
classifier_knnvis= svm.SVC(kernel='linear')
classifier_knnvis.fit(X_train[:,[0,1]], y_train)

# Visualising the Test set results of K-NN
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier_knnvis.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('SVM linear (Test set)')
plt.xlabel('Sepal Length(in cm)')
plt.ylabel('Sepal Width(in cm)')
L=plt.legend()
L.get_texts()[0].set_text('Setosa')
L.get_texts()[1].set_text('Versicolor')
L.get_texts()[2].set_text('Virginica')
plt.show()

'''
--------------------SVM RBF---------------
'''
svmp=svm.SVC(kernel='rbf')
svmp.fit(X_train,y_train)
accuracy=svmp.score(X_test,y_test)
print('svm rbf accuracy',accuracy)

#predicting 
svmppred=svmp.predict(X_test)
#confusion matrix
svmpconfusion=confusion_matrix(y_test,svmppred)

#Preparing another classifier for the Visualisation that trains the Classifier on Sepal characteristics
classifier_knnvis= svm.SVC(kernel='rbf')
classifier_knnvis.fit(X_train[:,[0,1]], y_train)

# Visualising the Test set results of K-NN
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier_knnvis.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Svm rbf (Test set)')
plt.xlabel('Sepal Length(in cm)')
plt.ylabel('Sepal Width(in cm)')
L=plt.legend()
L.get_texts()[0].set_text('Setosa')
L.get_texts()[1].set_text('Versicolor')
L.get_texts()[2].set_text('Virginica')
plt.show()

'''
--------------------------Naive's Bayes------------
'''

baye=naive_bayes.GaussianNB()
baye.fit(X_train,y_train)
accuracy=baye.score(X_test,y_test)
print('accuracy of naives bayes',accuracy)

#predicting 
bayepred=baye.predict(X_test)
#confusion matrix
bayeconfusion=confusion_matrix(y_test,bayepred)

#Preparing another classifier for the Visualisation that trains the Classifier on Sepal characteristics
classifier_knnvis= naive_bayes.GaussianNB()
classifier_knnvis.fit(X_train[:,[0,1]], y_train)

# Visualising the Test set results of K-NN
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier_knnvis.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Naives bayes (Test set)')
plt.xlabel('Sepal Length(in cm)')
plt.ylabel('Sepal Width(in cm)')
L=plt.legend()
L.get_texts()[0].set_text('Setosa')
L.get_texts()[1].set_text('Versicolor')
L.get_texts()[2].set_text('Virginica')
plt.show()
'''
----------------------Linear Regression-----------
'''

###Linear=linear_model.LinearRegression(n_jobs=-1)
##Linear.fit(X_train,y_train)
##accuracy=Linear.score(X_test,y_test)
##print(accuracy)
##
###predicting 
##linearpred=Linear.predict(X_test)
###confusion matrix
###linearconfusion=confusion_matrix(y_test,linearpred)
###Preparing another classifier for the Visualisation that trains the Classifier on Sepal characteristics
##classifier_knnvis= linear_model.LinearRegression(n_jobs=-1)
##classifier_knnvis.fit(X_train[:,[0,1]], y_train)
##
### Visualising the Test set results of K-NN
##X_set, y_set = X_test, y_test
##X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
##                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
##plt.contourf(X1, X2, classifier_knnvis.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
##            alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
##plt.xlim(X1.min(), X1.max())
##plt.ylim(X2.min(), X2.max())
##for i, j in enumerate(np.unique(y_set)):
##    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
##                c = ListedColormap(('red', 'green','blue'))(i), label = j)
##plt.title('K-Nearest Neighbours Classification (Test set)')
##plt.xlabel('Sepal Length(in cm)')
##plt.ylabel('Sepal Width(in cm)')
##L=plt.legend()
##L.get_texts()[0].set_text('Setosa')
##L.get_texts()[1].set_text('Versicolor')
##L.get_texts()[2].set_text('Virginica')
##plt.show()

'''
------------------Random Forest Classifier----------
'''

ranclass=ensemble.RandomForestClassifier(n_estimators=25)
ranclass.fit(X_train,y_train)
accuracy=ranclass.score(X_test,y_test)
print('random forest classifier accuracy',accuracy)

#predicting 
Randompred=ranclass.predict(X_test)
#confusion matrix
Randomconfusion=confusion_matrix(y_test,Randompred)

#Preparing another classifier for the Visualisation that trains the Classifier on Sepal characteristics
classifier_knnvis= ensemble.RandomForestClassifier(n_estimators=100)
classifier_knnvis.fit(X_train[:,[0,1]], y_train)

# Visualising the Test set results of K-NN
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier_knnvis.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Random forest Classification (Test set)')
plt.xlabel('Sepal Length(in cm)')
plt.ylabel('Sepal Width(in cm)')
L=plt.legend()
L.get_texts()[0].set_text('Setosa')
L.get_texts()[1].set_text('Versicolor')
L.get_texts()[2].set_text('Virginica')
plt.show()

'''
----------------------ACCURACY OF CLASSIFIERS---------------
'''
from sklearn.metrics import accuracy_score
knn_acc = accuracy_score(y_test,knnpred)
nb_acc = accuracy_score(y_test,bayepred)
svm_acc = accuracy_score(y_test,svmpred)
svmp_acc = accuracy_score(y_test,svmppred)
rf_acc = accuracy_score(y_test,Randompred)
#lin_acc=accuracy_score(y_test,linearpred)
barlist=plt.bar([1,2,3,4,5],height=[knn_acc, nb_acc, svm_acc, svmp_acc,rf_acc])
plt.xticks([1.45,2.45,3.45,4.45,5.45], ['K-Nearest\nNeighbours','Naive\nBayes','Support\nVector\nMachine',
           'Kernel\nSupport\nVector\nMachine','Random\nForest'])
barlist[0].set_color('r')
barlist[1].set_color('b')
barlist[2].set_color('g')
barlist[3].set_color('y')
barlist[4].set_color('k')
#barlist[5].set_color('c')
plt.xlabel('Type Of Classification')
plt.ylabel('Accuracy of Classifier')
plt.title('ACCURACY OF THE IMPLEMENTED CLASSIFIERS')
plt.show()


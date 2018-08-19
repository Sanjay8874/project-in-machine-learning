import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.naive_bayes import GaussainNB
style.use('ggplot')
df=pd.read_csv('E:\study\machine learning\Files\Breast-cancer-wisconsin.data.txt')

df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)
print(df.head())

X=np.array(df.drop(['class'],1))
print(X)
y=np.array(df['class'])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
clf=neighbors.KNearestClassifier()

clf.fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,10,10,3,2,3,3,2]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)




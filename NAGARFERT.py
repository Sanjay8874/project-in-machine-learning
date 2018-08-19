# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 20:47:24 2018

@author: UJJWAL
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing,svm,linear_model
from matplotlib import style
import matplotlib.pyplot as plt
import math
import datetime

style.use('ggplot')

df=pd.read_csv('NAGARFERT.csv')
df=df.drop(['Adj Close'],1)
df['hl_pct']=(df['High']-df['Close'])/df['Close']
df['pct_change']=(df['Close']-df['Open'])/df['Open']
df=df[['Close','hl_pct','pct_change','Volume']]
df.fillna(value=-99999,inplace=True)
df.replace(0,df.mean(),inplace=True)
forecast_col="Close"

forecast_out=int(math.ceil(0.001*len(df)))
df['label']=df[forecast_col].shift(-forecast_out)
X=np.array(df.drop(['label'],1))
X=preprocessing.scale(X)
X=X[:-forecast_out]
X_lately=X[-forecast_out:]


df.dropna(inplace=True)
y=np.array(df['label'])
#y = y[:-forecast_out]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
##clf=linear_model.LinearRegression(n_jobs=-1)
##clf.fit(x_train,y_train)
##accuracy=clf.score(x_test,y_test)
##print(accuracy)
##forecast_set=clf.predict(X_lately)
##print(-forecast_set/100,accuracy,forecast_out)


clf1=svm.SVR(kernel='linear')
clf1.fit(x_train,y_train)
accuracy=clf1.score(x_test,y_test)
print(accuracy)
##df['forecast']=np.nan
##last_date=df.iloc[-1].name
##last_unix=last_date.timestamp()
##one_day=86400
##next_unix=last_unix+one_day
##
##for i in forecast_set:
##    next_date=datetime.datetime.fromtimestamp(next_unix)
##    next_unix+=one_day
##    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i]
##
##df["Adj. Close"].plot()
##df["forecast"].plot()
##plt.legend(loc=4)
##plt.xlabel("Date")
##plt.ylabel("price")
##plt.show()




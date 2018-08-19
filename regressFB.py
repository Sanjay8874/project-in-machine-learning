import quandl
import math
import matplotlib.pyplot as plt
from matplotlib import style
from datetime import datetime
from sample import *
from sklearn.model_selection import train_test_split
from sklearn import preprocessing,svm
style.use('ggplot')

df=quandl.get('WIKI/FB',authtoken='XSmW6zzd9bLLJb8mnj6J')
#print(df.head())
df=df[['Open','High','Close','Volume','Low']]
#print(df.head())
df['pct_change']=(df['Close']-df['Open'])/df['Open']*100
df['hl_pct']=(df['High']-df['Low'])/df['Low']*100
df=df[['Open','Close','pct_change','hl_pct','Volume']]
forecast_col='Close'
df.fillna(value=-99999,inplace=True)
forecast_out=int(math.ceil(0.01*len(df)))
df['label']=df[forecast_col].shift(-forecast_out)
print(df['label'].head())

X=np.array(df.drop(['label'],1))
X=preprocessing.scale(X)
X=X[:-forecast_out:]
X_lately=X[-forecast_out:]

df.dropna(inplace=True)
y=np.array(df['label'])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=3283)
clf=svm.SVR(kernel='linear',epsilon=0.5)
clf.fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)
forecast_set=clf.predict(X_lately)
df['forecast']=np.nan
last_date=df.iloc[-1].name
last_unix=last_date.timestamp()
one_day=86400
next_unix=last_unix+one_day

for i in forecast_set:
    next_date=datetime.fromtimestamp(next_unix)
    next_unix+=one_day
    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i]
x=np.array(df['Close'],dtype=np.float64)
y=np.array(df['label'],dtype=np.float64)
regress(x,y)
df['Close'].plot()
df['forecast'].plot()
plt.legend(loc=4)
plt.show()






# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 19:31:49 2022

@author: vikas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('25_LogR/binary.csv')

df= df.dropna()
df


X = df['gre'].values.reshape((-1,1))

Y = df['admit'].values

plt.scatter(X,Y)


from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X,Y)

ypred = model.predict(X)

plt.scatter(X,Y)
plt.scatter(X,ypred)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X,Y)

ypred = model.predict(X)

ypred





# Case Bakery

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('25_LogR/bakery.csv')

df= df.dropna()
df


from sklearn.preprocessing import LabelEncoder

lab = LabelEncoder()

df['promotion'] = lab.fit_transform(df['promotion'])


X = df.drop(['promotion'], axis=1).values

Y = df['promotion'].values




from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X,Y)

ypred = model.predict(X)

ypred


from sklearn.metrics import accuracy_score, classification_report

accuracy_score(Y,ypred)

print(classification_report(Y,ypred))


y = [1,0,1,1,0,1,0,0,1,1]
yp= [1,0,1,1,0,1,0,0,1,1]

TP = 6
TN = 4
FP = 0
FN = 0

(TP +TN) / (TP+TN+FP+FN)


y = [1,1,1,1,0,1,0,0,1,1]
yp= [1,1,1,0,0,1,1,1,1,1]

TP = 6
TN = 1
FP = 2
FN = 1

(TP +TN) / (TP+TN+FP+FN)
accuracy_score(y,yp)

# Accuracy is not only parameter

y = [1,1,1,0,0,1,1,1,1,1]
yp= [1,1,1,1,1,1,1,1,1,1]

TP = 8
TN = 0
FP = 2
FN = 0

(TP +TN) / (TP+TN+FP+FN)


from sklearn.metrics import classification_report

print(classification_report(y,yp))




df['promotion'].value_counts()

# Class Unbalance

X
Y

from imblearn.over_sampling import SMOTE
oversample = SMOTE()
Xs, Ys = oversample.fit_resample(X, Y)

mean, meadian, sd

Xs

pd.Series(Ys).value_counts()

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(Xs,Ys)

ypreds = model.predict(Xs)

ypredprob = model.predict_proba(Xs) 


from sklearn.metrics import accuracy_score, classification_report

accuracy_score(Ys,ypreds)

print(classification_report(Ys,ypreds))





#Titanic


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('25_LogR/titanic.csv')

df= df.dropna()
df

X = df.drop(['survived'], axis=1).values
Y= df['survived'].values

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X,Y)

ypred = model.predict(X)
ypred

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(Y,ypred))

print(classification_report(Y,ypred))

df['survived'].value_counts()


from imblearn.over_sampling import SMOTE
oversample = SMOTE()
Xs, Ys = oversample.fit_resample(X, Y)




from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(Xs,Ys)

ypred = model.predict(Xs)
ypred

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(Ys,ypred))

print(classification_report(Ys,ypred))

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(Xs,Ys, test_size=0.4)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(xtrain,ytrain)

ypred = model.predict(xtest)
ypred

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(ytest,ypred))

print(classification_report(ytest,ypred))



# Decision Tree Classifiction

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(xtrain,ytrain)

ypred = model.predict(xtest)
ypred

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(ytest,ypred))

print(classification_report(ytest,ypred))


#Random Forest Classification

from sklearn.ensemble import RandomForestClassifier

model =  RandomForestClassifier()
model.fit(xtrain,ytrain)

ypred = model.predict(xtest)
ypred

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(ytest,ypred))

print(classification_report(ytest,ypred))


#Extra Tree Classifier

from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(xtrain,ytrain)

ypred = model.predict(xtest)
ypred

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(ytest,ypred))

print(classification_report(ytest,ypred))


#IRIS Case Study

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('25_LogR/iris.csv')

df

X= df.drop(['name'], axis=1).values

Y = df['name'].values



from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X,Y, test_size=0.4)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(xtrain,ytrain)

ypred = model.predict(xtest)
ypred

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(ytest,ypred))

print(classification_report(ytest,ypred))



# Decision Tree Classifiction

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(xtrain,ytrain)

ypred = model.predict(xtest)
ypred

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(ytest,ypred))

print(classification_report(ytest,ypred))


#Random Forest Classification

from sklearn.ensemble import RandomForestClassifier

model =  RandomForestClassifier()
model.fit(xtrain,ytrain)

ypred = model.predict(xtest)
ypred

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(ytest,ypred))

print(classification_report(ytest,ypred))


#Extra Tree Classifier

from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(xtrain,ytrain)

ypred = model.predict(xtest)
ypred

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(ytest,ypred))

print(classification_report(ytest,ypred))




# Clustering Algorithms

from sklearn.cluster import KMeans

data = {'x1': [25,34,22,27,33,33,31, 22,35,34,67,54,57,43,50,57,59,52,65, 47,49,48,35,33,44,45,38,43,51,46],'x2': [79,51,53,78,59,74,73,57,69,75,51,32, 40,47,53,36,35,58, 59,50,25,20,14,12,20,5,29,27,8,7]       }
  
df = pd.DataFrame(data,columns=['x1','x2'])
print (df)

import matplotlib.pyplot as plt

plt.scatter(df['x1'], df['x2'])

kmeans = KMeans(n_clusters=2)

kmeans.fit(df)

centroids = kmeans.cluster_centers_

centroids

df ['cluster'] = kmeans.labels_.astype(float)


plt.scatter(df['x1'], df['x2'], c= kmeans.labels_.astype(float), s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=80, marker="*")
plt.show()





kmeans = KMeans(n_clusters=10)

kmeans.fit(df)

centroids = kmeans.cluster_centers_

plt.scatter(df['x1'], df['x2'], c= kmeans.labels_.astype(float), s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=80, marker="*")
plt.show()



kmeans.inertia_



sse=[]

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)
sse

plt.style.use('fivethirtyeight')
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel('No of clusters')
plt.ylabel('SSE')
plt.show();



from kneed import KneeLocator
kl = KneeLocator(x=range(1,11), y=sse, curve='convex', direction='decreasing')
kl.elbow

kmeans = KMeans(n_clusters=3)

kmeans.fit(df)

centroids = kmeans.cluster_centers_

plt.scatter(df['x1'], df['x2'], c= kmeans.labels_.astype(float), s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=80, marker="*")
plt.show()



# Case Food Cluster

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('28_Clustering/FoodCluster.csv')
df


sse=[]

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)
sse

plt.style.use('fivethirtyeight')
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel('No of clusters')
plt.ylabel('SSE')
plt.show();



from kneed import KneeLocator
kl = KneeLocator(x=range(1,11), y=sse, curve='convex', direction='decreasing')
kl.elbow


kmeans = KMeans(n_clusters=3)
kmeans.fit(df)
centroids = kmeans.cluster_centers_

df['labels'] = kmeans.labels_

df.to_csv('FoodLabels.csv')



# Income
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('28_Clustering/income.csv')
df


sse=[]

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)
sse

plt.style.use('fivethirtyeight')
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel('No of clusters')
plt.ylabel('SSE')
plt.show();



from kneed import KneeLocator
kl = KneeLocator(x=range(1,11), y=sse, curve='convex', direction='decreasing')
kl.elbow


kmeans = KMeans(n_clusters=4)
kmeans.fit(df)
centroids = kmeans.cluster_centers_

df['labels'] = kmeans.labels_

df.to_csv('income.csv')




# Loan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('28_Clustering/loan_label.csv')
df


sse=[]

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)
sse

plt.style.use('fivethirtyeight')
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel('No of clusters')
plt.ylabel('SSE')
plt.show();



from kneed import KneeLocator
kl = KneeLocator(x=range(1,11), y=sse, curve='convex', direction='decreasing')
kl.elbow


kmeans = KMeans(n_clusters=3)
kmeans.fit(df)
centroids = kmeans.cluster_centers_

df['labels'] = kmeans.labels_

df.to_csv('loan_label.csv')



#MTCARS 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('28_Clustering/mtcarslables.csv')
df.columns

df = df.drop(['Unnamed: 0','labels', 'Unnamed: 13'], axis=1)
df.columns

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
DFs = SS.fit_transform(df)
DFs

sse=[]

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(DFs)
    sse.append(kmeans.inertia_)
sse

plt.style.use('fivethirtyeight')
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel('No of clusters')
plt.ylabel('SSE')
plt.show();



from kneed import KneeLocator
kl = KneeLocator(x=range(1,11), y=sse, curve='convex', direction='decreasing')
kl.elbow


kmeans = KMeans(n_clusters=3)
kmeans.fit(DFs)
centroids = kmeans.cluster_centers_

df['labels'] = kmeans.labels_

df.to_csv('mt_label.csv')




#Ames

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('ames.csv')

df.select_dtypes( 'object' )

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()

for col in df.select_dtypes( 'object' ).columns:
    df[col] = lab.fit_transform(df[col])

df



from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
DFs = SS.fit_transform(df)
DFs

sse=[]

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(DFs)
    sse.append(kmeans.inertia_)
sse

plt.style.use('fivethirtyeight')
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel('No of clusters')
plt.ylabel('SSE')
plt.show();



from kneed import KneeLocator
kl = KneeLocator(x=range(1,11), y=sse, curve='convex', direction='decreasing')
kl.elbow


kmeans = KMeans(n_clusters=4)
kmeans.fit(DFs)
centroids = kmeans.cluster_centers_

df['labels'] = kmeans.labels_

df.to_csv('Ames_label.csv')
















































































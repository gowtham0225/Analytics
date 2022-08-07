# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 19:28:56 2022

@author: vikas
"""


def minlist(l1):
    m=0
    for i in l1:
        if(m<i):
            m=i
    return(m)


def minlist(l1):
    m=l1[0]
    for i in l1:
        if(m>i):
            m=i
    return(m)


#Visual Analytics

import matplotlib.pyplot as plt

Year = [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010]
Unemployment_Rate = [9.8,12,8,7.2,6.9,7,6.5,6.2,5.5,6.3]
Unemployment_Rate1 = [1.8,2,8,4.2,6.0,7,3.5,5.2,7.5,5.3]



plt.plot(Year, Unemployment_Rate )
plt.title('UR vs Year')
plt.xlabel("Year")
plt.ylabel("UR")
plt.show()


['r','b','k','o','y']
plt.plot(Year, Unemployment_Rate , marker='o', color='r',markersize=10, label='UR1')
plt.plot(Year, Unemployment_Rate1, marker = '*', color='y', markersize=10, label='UR2')
plt.title('UR vs Year')
plt.xlabel("Year")
plt.ylabel("UR")
plt.legend()
plt.show()



col = ['#F7EDDB', '#FFAABB', '#009FFD','#276F98']

plt.plot(Year, Unemployment_Rate , marker='o', color=col[3] ,markersize=10, label='UR1')
plt.plot(Year, Unemployment_Rate1, marker = '*', color=col[2], markersize=10, label='UR2')
plt.title('UR vs Year')
plt.xlabel("Year")
plt.ylabel("UR")
plt.legend()
plt.show()






import pandas as pd
Data = {'Year': [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010], 'Unemployment_Rate': [9.8,12,8,7.2,6.9,7,6.5,6.2,5.5,6.3]}
Data  
df = pd.DataFrame(Data,columns=['Year','Unemployment_Rate'])
df  

plt.plot(df['Year'], df['Unemployment_Rate'] , marker='o', color=col[3] ,markersize=10, label='UR1')
plt.title('UR vs Year')
plt.xlabel("Year")
plt.ylabel("UR")
plt.legend()
plt.show()


col = ['r','b','k','y']

Country = ['USA','Canada','Germany','UK','France']
GDP_Per_Capita = [45000,42000,52000,49000,47000]

plt.bar(Country,GDP_Per_Capita, color=col)
plt.title('Country vs GDP_Per_Capita', fontsize=16)
plt.xlabel('Country',fontsize=14)
plt.ylabel('GDP_Per_Capita', fontsize=14)
plt.grid(True)


import matplotlib.pyplot as plt
from pydataset import data
mt = data('mtcars')

plt.scatter(mt['mpg'], mt['hp'], c=mt['gear'])
plt.xlabel('MPG')
plt.ylabel("HP")
plt.title("MPG vs HP")



import matplotlib.pyplot as plt
import seaborn as sns

sns.set() #default settings

tips_df = sns.load_dataset('tips')
tips_df

tips_df.columns
tips_df.total_bill

total_bill = tips_df.total_bill.to_numpy()


tip = tips_df.tip.to_numpy()


tip

plt.scatter(total_bill, tip)
plt.show();



#Subplots

import matplotlib.pyplot as plt


fig, ax = plt.subplots(2,1)
ax[0].bar(['CS','EC','ME'], [10,20,30])
ax[1].scatter(total_bill, tip)



# RC Factors

import matplotlib.pyplot as plt

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


Year = [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010]
Unemployment_Rate = [9.8,12,8,7.2,6.9,7,6.5,6.2,5.5,6.3]
Unemployment_Rate1 = [1.8,2,8,4.2,6.0,7,3.5,5.2,7.5,5.3]

plt.plot(Year, Unemployment_Rate )
plt.title('UR vs Year', fontsize=18)
plt.xlabel("Year")
plt.ylabel("UR")
plt.show()





import numpy as np
import matplotlib.pyplot as plt


d1 = np.random.normal(100, 10, 200)

d1

plt.hist(d1)



fig = plt.figure(figsize =(5, 5), dpi=300) 
plt.boxplot(d1)

np.mean(d1)
np.median(d1)
np.std(d1)
np.min(d1)
np.max(d1)




import seaborn as sns
sns.set_theme(style="whitegrid")
tips = sns.load_dataset("tips")
tips.columns

ax = sns.boxplot(x=tips["total_bill"])
ax = sns.boxplot(x=tips["tip"])


d1=tips["total_bill"]

np.mean(d1)
np.median(d1)
np.std(d1)
np.min(d1)
np.max(d1)

plt.hist(d1)


d2 = d1[d1<=40]
d2

plt.hist(d2)
np.mean(d2)
np.median(d2)
np.std(d2)
np.min(d2)
np.max(d2)



sizes=[120,30,10]
labels = ['BBA', 'MBA','PHD']

fig1 = plt.figure(figsize =(5, 5), dpi=300)
plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False)
plt.show()




# Simple Linear Regression

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = np.array([5,15,25,35,45,55])
x.shape


x = x.reshape((-1,1))  #making 2 dim
x.shape

y = np.array([5,20,14,32,22,38])  #1 dim
y.shape


plt.scatter(x,y)


model = LinearRegression()

model.fit(x,y)

ypred = model.predict(x)
ypred

plt.scatter(x,y)
plt.scatter(x,ypred)


r2 = model.score(x,y)

# R2 always lies between 0 to 1




# MTDACRS

from pydataset import data

mt = data('mtcars')

mt.columns


X = mt['mpg'].values.reshape((-1,1))
X.shape

Y = mt['hp'].values


plt.scatter(X,Y)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X, Y)

r2 = model.score(X,Y)
r2

ypred = model.predict(X)

plt.scatter(X,Y)
plt.scatter(X,ypred)


from sklearn.metrics import r2_score

r2 = r2_score(Y, ypred)

r2

X.shape

xpred = np.array([27, 39, 12]).reshape((-1,1))
xpred.shape
xpred

ypred1 = model.predict(xpred)

ypred1



plt.scatter(X,Y)
plt.scatter(X,ypred)
plt.scatter(xpred,ypred1, marker='*')


#Case Housing


import pandas as pd

df = pd.read_csv('24_LR/Housing.csv')

df.columns

df.describe()

df = df.dropna()

df

X = df['area'].values.reshape((-1,1))

Y = df['price'].values

plt.scatter(X,Y)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X,Y)

r2 = model.score(X,Y)
r2




#Case


import pandas as pd

df = pd.read_csv('24_LR/data.csv')

df.columns

df = df.dropna()


X = df['Engine HP'].values.reshape((-1,1))
Y = df['MSRP'].values



plt.scatter(X,Y)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X,Y)

r2 = model.score(X,Y)
r2



#Multi Linear Regression 

mt.dtypes

X = mt[['cyl','disp','drat','wt', 'hp']].values
X.shape

Y = mt['mpg']
Y.shape


from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X,Y)

r2 = model.score(X,Y)
r2


import matplotlib.pyplot as plt
from pydataset import data
mt = data('mtcars')

import statsmodels.api as sm
from statsmodels.formula.api import ols
MTmodel1 = ols("mpg ~ cyl + disp + hp + drat + wt + qsec + vs+ am+ gear+carb", data=mt).fit()

MTmodel1.summary()

mt.columns

plt.hist(mt['mpg'])


fig = plt.figure(figsize=(12, 8))
fig = sm.graphics.plot_ccpr_grid(MTmodel1, fig=fig)



#Case Life Expectancy Data.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('24_LR/Life Expectancy Data.csv')

df.columns
df = df.dropna()


Y = df['Life expectancy '].values

X = df.drop(['Country', 'Year', 'Status', 'Life expectancy '], axis=1).values

X.shape

Y.shape


from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X,Y)

r2 = model.score(X,Y)

r2


from sklearn.model_selection import train_test_split

Xtrain, Xtest,Ytrain, Ytest = train_test_split(X,Y, test_size=0.1 )


model.fit(Xtrain,Ytrain)

ypred = model.predict(Xtest)

from sklearn.metrics import r2_score
r2_score(Ytest, ypred)




col = ['Adult Mortality',
       'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
       'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure',
       'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',
       ' thinness  1-19 years', ' thinness 5-9 years',
       'Income composition of resources', 'Schooling']

st=''
for i in col:
    st= st+' + '+ i
    
    
df.columns


df = pd.read_csv('24_LR/Life Expectancy Data.csv')

df =df.dropna()


df = df.drop(['Country', 'Year', 'Status'], axis=1)


for i in df.columns:
    print(len(df[i]))


len(df.columns)

lst = []
for i in range(1,20):
    lst.append('A'+str(i))

df.columns = lst
lst
len(df.columns)

len(df)

import statsmodels.api as sm
from statsmodels.formula.api import ols
MTmodel1 = ols("A1 ~ A2 + A3 + A4 + A5 + A6 +A7 + A8 ", data=df).fit()

MTmodel1.summary()

mt.columns

plt.hist(mt['mpg'])


fig = plt.figure(figsize=(12, 8))
fig = sm.graphics.plot_ccpr_grid(MTmodel1, fig=fig)

plt.boxplot(df['A2'])

df = df[df['A2']<430]

































































































































































































































































































































































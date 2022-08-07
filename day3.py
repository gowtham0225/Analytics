# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 19:31:42 2022

@author: vikas
"""

import pandas as pd

from pydataset import data

pd.set_option('display.max_rows', 500)
print(data(''))

'''
df = data('')
df.to_csv('data.csv')
'''

df = data('mtcars')
type(df)
df
df.head()
df.tail()

df.columns
df.index

type(df.index)

df.head()

df.to_csv('mt.csv')


df1 = pd.read_csv("mt.csv")
df1

df2 = pd.read_csv(r'D:\ML-Lab\Dataset\brain_size.csv', sep=';')
df2

type(df1)



s = pd.Series([4,3,1,5,6])
s

type(s)


s1 = pd.Series(['a','b','c','d','e'], index=[2,4,6,5,8])
s1

s2 = pd.Series(['a','b','c','d','e'], index=[2,4,2,5,8])
s2
s2[2]
s2.loc[2]

#Displaying Index Value, and System Index Value

s2.iloc[2]


'''
l1 = [2,3,5,3,5]
l1[2]
'''

import numpy as np

n1 = np.random.randint(1, 100, size=100)
n1

s3 = pd.Series(n1) 
s3

s3>50
s3[s3>80]

s3[(s3>80) & (s3<90)]

import pandas as pd
import numpy as np


course = pd.Series(['BTech','MTech','BBA','MBA'])
course
strength = pd.Series([100, 50, 200, 75])
strength
fees = pd.Series([2.5, 3, 2, 4])
fees

d1 = {'Course':course, 'Strength':strength, 'Fees':fees}
df = pd.DataFrame(d1)
df

df.values
df.columns

df['Course']

df1 = df[['Course','Strength']]

df
df.loc[1]
df.loc[1:2]
df.iloc[1:3]

df.Course == 'MTech'

df[df.Course == 'MTech']

df

df.Strength >=100

df[df.Strength >=100]


emp = pd.read_csv('employees.csv')
emp.head()

emp.columns


emp[emp.SALARY >=10000]

emp.describe()
emp.count()



import numpy as np
import pandas as pd

rno = pd.Series(np.arange(1, 101))
rno

name = []

for i in range(1, 101):
    name.append('Student'+str(i))
    
name = pd.Series(name)
name

course = pd.Series(np.random.choice(['MBA', 'BBA', 'BTech', 'MTech'], size=100))
course

campus = pd.Series(np.random.choice(['DL','MB','CH', 'KO'], size=100))
campus

gender = pd.Series(np.random.choice(['M','F'], size=100))
gender


df = pd.DataFrame({'Rno':rno,'Name':name, 'Course':course, 
                   'Campus':campus, 'Gender':gender})
df


df = pd.DataFrame({'Rno':rno,'Name':name, 'Course':course, 
                   'Campus':campus, 'Gender':gender})
df


df = df.set_index('Rno')
df


courseDetail = pd.Series(['MBA', 'BBA', 'BTech', 'MTech'])

fees = pd.Series(['1L','1.5L','1.7L','1.2L'])


Fees = pd.DataFrame({'Course':courseDetail, 'Fee':fees})

Fees

pd.merge(df, Fees)



#! Million Dataset


import numpy as np
import pandas as pd

rno = pd.Series(np.arange(1, 100001))
rno

name = []

for i in range(1, 100001):
    name.append('Student'+str(i))
    
name = pd.Series(name)
name

course = pd.Series(np.random.choice(['MBA', 'BBA', 'BTech', 'MTech'], size=100000))
course

campus = pd.Series(np.random.choice(['DL','MB','CH', 'KO'], size=100000))
campus

gender = pd.Series(np.random.choice(['M','F'], size=100000))
gender


df = pd.DataFrame({'Rno':rno,'Name':name, 'Course':course, 
                   'Campus':campus, 'Gender':gender})
df

df.to_csv('Student.csv')



#Data Cleaning

pd4 = pd.DataFrame([['dhiraj', 50, 'M', 10000, None], ['Vikas', None, None, 'aa', None], ['kanika', 28, None, 5000, None], ['tanvi', 20, 'F', None, None], ['poonam',45,'F',100000,True],['upen',None,'M',None, None]])
pd4


pd4.dropna()
pd4.dropna(axis=0)
pd4.dropna(axis='rows')

pd4
pd4.dropna(axis=1)
pd4.dropna(axis='columns')


pd4.dropna(axis=1, how='all')

pd4.dropna(axis=1, how='any')

pd4.dropna(axis=0, thresh = 2)
pd4

pd4.fillna(0)



df = pd.read_csv('airline.csv')
df

df.plot()

df.fillna(0).plot()

df[:30].plot()
df[:30].fillna(0).plot()

df[:30].fillna(method='ffill').plot()

df[:10]

df[:10].fillna(method='ffill')
df[:10].fillna(method='bfill')


grades1 = {'subject1': ['A1','B1','A2','A3'],'subject2': ['A2','A1','B2','B3']   }
df1 = pd.DataFrame(grades1)
df1

grades2 = {'subject1': ['A1','B1','A2','A3'],'subject4': ['A2','A1','B2','B3']}
df2 = pd.DataFrame(grades2)
df2

df1
df2

df3 = pd.concat([df1,df2])
df3

df4 = pd.concat([df1,df2], axis=1)
df4


import pandas as pd
#Join
rollno = pd.Series(range(1,11))
rollno

[ "Student" + str(i) for i in range(1,11)]

list(range(1,11))


name = pd.Series(["student" + str(i) for i in range(1,11)])
name

genderlist  = ['M','F']

import random

gender = random.choices(genderlist, k=10)
gender

random.choices(population=genderlist,weights=[0.4, 0.6],k=10)

import numpy as np
#numpy.random.choice(items, trials, p=probs)
np.random.choice(a=genderlist, size=10, p=[.2,.8])


import numpy as np
marks1 = np.random.randint(40,100,size=10)
marks1


pd5 = pd.DataFrame({'rollno':rollno, 'name':name, 'gender':gender})

pd5




rollno1 = pd.Series(range(6,16))
rollno1

#course = random.choices( population=['BBA','MBA','BTECH'] ,weights=[0.4, 0.3,0.3],k=10)
course = np.random.choice(a=['BBA','MBA','BTECH'], size=10)
course

marks2 = np.random.randint(40,100,size=10)

marks2

pd6 = pd.DataFrame({'rollno':rollno1, 'course':course, 'marks2':marks2})
pd6

pd5
pd6

pd.merge(pd5,pd6, how='outer')
pd.merge(pd5,pd6, how='inner')
pd.merge(pd5,pd6, how='left')
pd.merge(pd5,pd6, how='right')



name1 = pd.Series(["student" + str(i) for i in range(6,16)])
name1

pd7 = pd.DataFrame({'rollno1':rollno1, 'name1': name1, 'course':course, 'marks2':marks2})
pd7


pd.merge(pd5, pd7, left_on='rollno', right_on='rollno1')

pd.merge(pd5, pd7, left_on=['rollno','name'] , right_on=['rollno1', 'name1'])



import pandas as pd
import numpy as np

rollno = pd.Series(range(1,1001))
rollno

name = pd.Series(["student" + str(i) for i in range(1,1001)])
name

genderlist  = ['M','F']
gender= np.random.choice(a=genderlist, size=1000)
gender

marks1 = np.random.randint(40,100,size=1000)

marks2 = np.random.randint(40,100,size=1000)

fees = np.random.randint(50000,100000,size=1000)

course = np.random.choice(a=['BBA','MBA','BTECH', 'MTech'], size=1000)
course

city = np.random.choice(a=['Delhi', 'Gurugram','Noida','Faridabad'], size=1000, replace=True, p=[.4,.2,.2,.2])

pd8 = pd.DataFrame({'rollno':rollno, 'name':name, 'course':course, 'gender':gender, 'marks1':marks1,'marks2':marks2, 'fees':fees,'city':city})
pd8

pd8.head()


pd8.groupby('course').size()
pd8.groupby('course').count()


pd8.groupby(['course', 'gender']).size()
pd8.groupby(['course', 'city','gender']).size()


pd8.columns

df = pd8.groupby(['course', 'city','gender']).aggregate({'marks1':['size', np.mean, max, min ]})
df.to_csv('studentgroup.csv')


pd8
pd12 = pd8.pivot_table(index=['city','course'], columns='gender', aggfunc='size')


pd.crosstab([pd8.city],  pd8.course)


#Denco Case Study

df = pd.read_csv(r'D:\ML-Lab\Analytics2606\20denco.csv')

df.columns

df.count()
df.dtypes

df.groupby('custname').size().sort_values(ascending=False)

pd9 = df.groupby('custname').size().sort_values(ascending=False).head(10)
pd9.plot(kind='bar')



pd10 = df.groupby(['custname']).aggregate({'revenue':'sum'}).sort_values(ascending=False, by='revenue').head(10)
pd10.plot(kind='bar')


df.columns

pd11 = df.groupby(['partnum']).aggregate({'revenue':'sum'}).sort_values(ascending=False, by='revenue').head(10)
pd10.plot(kind='bar')


pd8

pd8.to_excel('data.xlsx')



with pd.ExcelWriter('data1.xlsx') as writer:
    pd8.to_excel(writer, sheet_name='first', index=False)
    pd8.to_excel(writer, sheet_name='second')
    pd12.to_excel(writer, sheet_name='pivot')



df = pd.read_excel('data1.xlsx', sheet_name='pivot')
df




#Lambda Function

def f(x):
    return(x**2)

f(2)
f(5)



fl = lambda x: x**2

fl(2)

fl(5)

fl1 = lambda x,y : x*y

fl1(4,7)

fl2 = lambda x: x.upper()

fl2('python')

pd8['name'] = pd8['name'].apply(fl2)
pd8['name']



l1 = [4,3,6,5,8,9]

sq = lambda x: x**2

l2= list(map(sq, l1))
l2


l1

fl3 = lambda x: x%2==0

l3 = list(filter(fl3, l1))
l3


l4 = [-1,2,5,-6,-3, 8,7,6,-9]

fl4 = lambda x: x>=0

l5 = list(filter(fl4, l4))
l5













































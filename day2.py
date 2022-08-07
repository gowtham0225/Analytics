# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 19:26:41 2022

@author: vikas
"""

# Dictionary
# Indexed or Not Indexed,Mutable/changable or Not Mutable,
#Ordered or not ordered, Hetrogeneous or Homogeneous,
# Uniqueness or not


import pandas as pd
d1 = {}

# Key value
d1 = {'rollno': 1, 'name': 'VK', 'Class': 'Analytics'}
d1

# Not Indexed

d1['rollno']


# Hetrogeneous
car = {'brand': 'Honda', 'model': 'Jazz', 'year': 2020}
car

#not ordered

car['brand']
car['year']

# Mutable/changable

car['brand'] = 'Maruti'
car

car['color'] = 'Black'
car


for i in car.keys():
    print(i)


for i in car.items():
    print(i)


car
car.pop('color')
car

# car.pop(['brand','model']) Not possible to delete multiple elements


car.get('brand')
car.popitem()
car

car.values()


rollno = list(range(1, 11))
rollno


name = []

for i in range(1, 11):
    name.append('Student'+str(i))

name

stud = {'RollNo': rollno, 'Name': name}
stud

df = pd.DataFrame(stud)
df


# Tuple

t1 = ()


t1 = (3, 2, 5, 7, 2)
t1

# Hetrogeneous

t2 = (33, 54.6, "ss", True)

t2


# indexed
t2
t2[2]
t2[5]  # IndexError: tuple index out of range

# Not Mutable


t2[2] = 33  # TypeError: 'tuple' object does not support item assignment

t2.pop()  # AttributeError: 'tuple' object has no attribute 'pop'

a = 2, 3


# Conditional Statements

# if statement

'''
if condition:
    statements
    statements
statment
    
    
if
{
 
 
 
}
'''

a = 20
b = 30

a < b

if a < b:
    print('b is greater')
    print('a is greater')


a = 20
b = 30

if a < b:
    print('b is greater')
    print('a is lesser')
else:
    print('a is greater')
    print('b is lesser')


marks = int(input('Enter Marks'))

marks
if (marks > 90):
    print("Grade A")
elif (marks > 80 and marks <= 90):
    print("Grade B")
elif(marks > 70 and marks <= 80):
    print("Grade C")
elif(marks > 60 and marks <= 70):
    print("Grade D")
else:
    print("Fail")


# Looping Statements or Iterative Statements
# For Loop and While Loop


l1 = [1, 5, 4, 7, 8]
l1

for i in l1:
    print(i*2)

for i in range(1, 100, 5):
    print(i)


for i in range(1, 11):
    print('2 * {0} = {1}'.format(i, i*2))


a = 20

print("value of a is {0}".format(a, b))


for j in range(2, 5):
    for i in range(1, 11):
        print('{0} * {1} = {2}'.format(j, i, i*j))


teamA = ['Australia', 'Pakistan', 'India',
         'England']   # 4elements   list index 0-3

for i in teamA:
    print(i)


for i in teamA:
    if(i == 'India'):
        print(i)
    else:
        print("Not India")


# While Loop


# conditional check

i = 1
while(i <= 10):
    print(i)
    i = i+1


i = 10
while(i >= 1):
    print(i)
    i = i-1


i = 1
while(i <= 10):
    print('2 * {0} = {1}'.format(i, i*2))
    i = i+1


j = 1
while(j <= 5):
    i = 1
    while(i <= 10):
        print('{0} * {1} = {2}'.format(j, i, i*j))
        i = i+1
    j = j+1


#Break, Continue


l1 = list(range(1, 101))

ele = 17
i = 0
while(True):
    print(l1[i])
    if(l1[i] == ele):
        print("Element Searched")
        break
    i = i+1


# Functions


a = 10
b = 20

print(a+b)
print(a*b)
print(a/b)


print(a+b)
print(a*b)
print(a/b)


print(a+b)
print(a*b)
print(a/b)


def oper():
    a = 20
    b = 20
    print(a+b)
    print(a*b)
    print(a/b)


oper()

oper()

oper()


def oper1(a, b):
    print(a+b)
    print(a*b)
    print(a/b)


oper1() #TypeError: oper1() missing 2 required positional arguments: 'a' and 'b'

oper1(10, 20)

oper1(30,4)

oper1(66,3)



def printhello(User):
    print("Hello ", User)


printhello("Amit")


printhello("SK")


def user(name, iden, batch, year):
    print(name, iden, batch, year)


user('Vk', 11, 'Analytics', 2022)


def user1(name,  batch, iden=1, year=2022):
    print(name,  batch, iden, year)

user1('Vk',  'Analytics',11, 2020)

user1('Vk', 'Analytics')

user1(batch=10, name='SK')


def maxlist(l1):
    m=0
    for i in l1:
        if(m<i):
            m=i
    return (m)
    

lst = [10,4,3,7,6,8,1]
maxlist(lst)

lst = [3,65,2,6,2,66,33,99]
maxlist(lst)

max(lst)


def evenodd(n):
    if(n%2==0):
        print("Even")
    else:
        print("Odd")


evenodd(29)

evenodd(2)



# Random Numbers
import random

random.randint(10,20)


import random as rd

rd.randint(10, 20)


lst = [11,22,66,55,77,44]
rd.choice(lst)


gen = ['M','F']

sample = rd.choices(gen,k=10)

sample

result = ['Pass', 'Fail']

sample = rd.choices(result, k=20)
sample


# Numpy

import numpy as np

x0 = np.random.randint(10,20)

type(x0)


x1 = np.random.randint(1,10,size=6)
x1.shape


x2 = np.random.randint(1,10, size=(3,4))
x2.shape

x3 = np.random.randint(1,10, size=(2,3,4))
x3.shape

#Access

x1
x1[0]
x1[2]
x1[0:3]
x1[2:]
x1[-1]
x1[-3:]


x2 = np.random.randint(1,10, size=(3,4))
x2.shape
x2

x2[0][0]
x2[1]
x2[1][2]

x2 = np.random.randint(1,10, size=(5,6))
x2.shape
x2

x2[2][2:4]

x2[3][-1]

x2[-1,-1]

x2
x2[2:4,2:4]


x3 = np.random.randint(1,10, size=(3, 5,6))
x3.shape
x3

x3[0:2]
x3[-1]

x3[-1,-1,]


import numpy as np

a1 = np.arange(20)
a1
a2 = np.arange(10,20)
a2
a3 = np.arange(10,50,5)
a3

a4 = np.random.randint(1, 10, size=(5,4))
a4
a4.shape

a5 = a4.reshape(4,5)
a5

a5 = a4.reshape(2,10)
a5


a6 = np.zeros((5,4))
a6

type(a6[0,0])


'''
bt = 1024*1024*1024*8
bt/64
64 
'''

a6 = np.ones((4,5))
a6


a7 = np.eye(3,3)
a7

a8 = np.linspace(0,10,6)
a8



a9 = np.random.randint(1, 10, size=5)

np.mean(a9)
np.std(a9)
np.var(a9)

a9 = np.random.randint(1, 10, size=(5,4))
a9
np.mean(a9)
np.std(a9)
np.var(a9)


a9 = np.random.randint(1, 10, size=(5,4))
a9
np.mean(a9, axis=0)
np.mean(a9, axis=1)
np.std(a9)
np.var(a9)


np.floor([1.2, 1.6])
np.ceil([1.2, 1.6])
np.trunc([1.2, 1.6])

np.floor([-1.2, -1.6])
np.ceil([-1.2, -1.6])
np.trunc([-1.2, -1.6])

np.round([1.4444, 3.5546543],2)

np.round([1.4444, 3.5546543])
np.round([1.4444, 3.5546543],4)



n1 = np.random.randint(1,10, size=(3,5))
n1

n2 = np.random.randint(20,30, size=(3,5))
n2

n3 = np.concatenate([n1,n2])
n3

n4 = np.concatenate([n1,n2],axis=1)
n4

np.split(n3, 2)
np.split(n4, 2, axis=0)

n4.shape
n4

n2

np.max(n2)
np.max(n2, axis=1)
np.min(n2)
np.median(n2)
np.sum(n2)
np.sum(n2, axis=0)

n2
n2>27

np.sum(n2>27)

np.sum(n2>26, axis=1)

np.sort(n2)




import matplotlib.pyplot as plt
import numpy as np

nor = np.random.normal(100, 2, size=10000)

nor
plt.hist(nor)


np.mean(nor)
np.median(nor)





























































































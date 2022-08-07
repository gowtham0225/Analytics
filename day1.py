# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 20:35:43 2022

@author: vikas
"""

a=10
b=20
c=a+b

print(c)


"Hi"
'hi'

"hi's"

'hi's


print("Hi")

print?
help(print)

print("Hi","Welcome","Python")

print("Hi","Welcome","Python", sep='-')

print("Hi")
print("Welcome")
print("Python")

print("Hi", end='---')
print("Welcome", end='-----')
print("Python")


import pandas
print(pandas.__version__)


a = 10
b = 20
c = 'abc'
d = 1.11
e= True

print (a,b,c,d,e)


"The value of a is {0}, and b is {1}".format(a,b)

print("The value of a is 10, and b is 20", sep='--')

print("The value of a is",a , "and",b,"is 20", sep='--')

print("The value of a is {0}, and b is {1}".format(a,b))


#index always start with 0
print("a","b","c")

a = 10

a = input("Enter your value")

b =20

c = a + b

a= "Vikas"
b = "Khullar"
c =  a + b

print(c)

a =10
b =20

c =a + b

print(a)

a = input("Enter a->")
b = input("Enter b->")

c=a+b
print(c)


#Type Conversion or Type Casting

a = "10"

type(a)

s_i = int(a)

b = "10"
type(b)
s_b = int(b)
s_b

s_i+s_b


a = "10"

sa = int(a)
fa = float(a)
fa


v =7

"Welcome to Python " + str(v)

print ()

# Demonstrate int() Casting Function

float_to_int = int(3.5)
string_to_int = int("1")
print(f"After Float to Integer Casting the result is {float_to_int}")
print(f"After String to Integer Casting the result is {string_to_int}")


# Demonstrate float() Casting Function

int_to_float = float(4)
string_to_float = float("1")
print(f"After Integer to Float Casting the result is {int_to_float}")
print(f"After String to Float Casting the result is {string_to_float}")

# Demonstrate str() Casting Function

int_to_string = str(8)
float_to_string = str(3.5)
print(f"After Integer to String Casting the result is {float_to_string}")
print(f"After Float to String Casting the result is {float_to_string}")


#Airthmatic Operations

x=9

print(x)

print(x+2)
print(x-2)
print(x*2)
print(x/2)

print(x%2)


a = int(input("Enter Value-> "))
if (a%2==0):
    print(f"{a} is Even Number")
else:
    print(f"{a} is Odd Number")


print(x**2)
print(x**4)
print(x**(1/2))
print(x**(1/3))


print(x)
x = x +2
print(x)
x = x**3
print(x)
y = x
y = x+5


#single line comment

'''
Multi
line comments
'''

#Boolean

t = True
f = False

'''
AND
a b O
0 0 0
0 1 0
1 0 0
1 1 1


OR
a b O
0 0 0
0 1 1
1 0 1
1 1 1
'''

t = True
f=False

print(f and f)
print(f and t)
print(t and f)
print(t and t)


print(f or f)
print(f or t)
print(t or f)
print(t or t)



#Logical or Comparison Operators
'''
== is equal to
<
>
<=
>=
!=
'''


a = 10
b = 20

#a = b

a==b
a<b
a>b
a<=b
a>=b
a!=b

a = 10
b = 20
c = 30

a<b and b>c

a<b or b>c

if (a<b and b<c):
    print("a is least")

not(f)


#Strings

fn = "Welcome"
tn = "Python"

name = fn + tn
name = fn + " " + tn

h = 'hello'

h = h.capitalize()
h = h.upper()
h =h.lower()
print(h)

s = " Welcome to Java"

s = s.replace("Java", "Python")
print(s)

name = input("Enter Name->")
name = name.strip()
name

'''
name = name.split(' ')
fname = name[0]
lname = name[1]
'''

# List, Set, Dictionary, Tuple

#Indexed or Not Indexed,Mutable/changable or Not Mutable,
#Ordered or not ordered, Hetrogeneous or Homogeneous, 
#Uniqueness or not


#List

l1 = []
l2 = [2,4,6,8,1]

print(l2)

#Unordered

print(l2)

#Indexed

l2[0]
l2[1]
l2[2]
l2[3]
l2[4]
l2[5]


r1 = range(10)
r1
l3 = list(r1)
l3

r2 = range(3, 30)
r2
l4 = list(r2)
l4

r3 = range(1,101, 5)
r3
l4 = list(r3)
l4

l5 = list(range(1, 10001))
print(l5)


for i in l2:
    print(i**2)

l2

#Mutable or Changable
l2[1] = 44
l2

# Hetrogeneous

l6 = [111, 'Python', True, 22.33]
l6



type(l6)

type(l6[0])
type(l6[1])
type(l6[2])
type(l6[3])


#Not Supporting uniqueness

l7 = [3,4,3,4,22,4]
l7

l4
l4[0:3]
l4[:3]
l4[5:8]
l4[8:]
l4[-1]
l4[-4:]
l4[-5:-2]


l6.append('10')

l6
l6.insert(2, 'Analytics')
l6

r = l6.pop()
print(r)
l6
r = l6.remove(22.33)
print(r)
l6

print(l6.append('10'))
print(l6)

l6.append('10')
print(l6)

l6 = l6.append('20')
print(l6)

l6

l6.remove(22.33)
l6

l6.remove(22.33)

print(l6)

l8 = [5,1,3,9,6,7]

l8.sort()
l8

l8.reverse()
l8

l5.reverse()
l5


#Set
#Indexed or Not Indexed,Mutable/changable or Not Mutable,
#Ordered or not ordered, Hetrogeneous or Homogeneous, 
#Uniqueness or not


s1 = {1}

#Maintain Uniqueness
s1 = {10, 20, 10, 40, 30, 20}
print(s1)
#Not Indexing
s1[3] #TypeError: 'set' object is not subscriptable


#Hetrogeneous

s2 = {30, 'Python', True, 44.3}
s2


s1

for i in s1:
    print(i)

#Ordered
s1

s1.add(11)
s1

s1.remove(20)
s1
s1.remove(20) #KeyError: 20
s1
s1.discard(30)
s1
s1.discard(30)
s1

a = s1.pop()
print(a)

s1.update([111,22])
s1
s1.add(33)
s1
s1.update([44,55,66,77])
s1



teamA = {'India', 'Australia','Pakistan', 'England'}
teamB = {'Bangladesh', 'New Zealand', 'West Indies', 'India'}


teamA
teamB


teamA.union(teamB)
teamA.intersection(teamB)
teamA.difference(teamB)


























































































































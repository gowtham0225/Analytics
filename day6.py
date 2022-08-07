# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 19:30:57 2022

@author: vikas
"""


from efficient_apriori import apriori

transactions = [['eggs', 'bacon', 'soup', 'milk'], 
                ['eggs', 'bacon', 'apple', 'milk'], 
                ['soup', 'bacon', 'banana']]


transactions

'''

eggs -> bacon

support
eggs to bacon  = 0.66


confidence
=1.0

lift = confidence/ y independance

1/(3/3) = 1


bacon -> eggs

support

2/3 =0.667

confidence = 2/3
0.667

lift

0.667/ (2/3) = 1

'''


itemsets, rules = apriori(transactions)

rules

for rule in rules:
    print(rule)
    
    
    

rules_1 = list(filter(lambda rule: len(rule.lhs) == 1 and len(rule.rhs) == 1, rules))


#rules_1

for rule1 in rules_1:
    print (rule1)
    

# MLXTend


import numpy as np
import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

transactions = [['milk', 'water'], ['milk', 'bread'], 
                ['milk','bread','water']]
transactions


te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
te_ary

df = pd.DataFrame(te_ary, columns=te.columns_)
df


itemsets = apriori(df, use_colnames=True)
itemsets
pd.set_option('display.max_columns',None)

rules = association_rules(itemsets,min_threshold=0.000001)
rdf = rules[['antecedents', 'consequents','support','confidence','lift']]


rdf [rdf ['confidence']>=1]


-
#Case Store

import pandas as pd
import numpy as np


store_data = pd.read_csv('29_Apriori/store_data1.csv', header=None)
store_data.head()

records = []

for i in range (0, len(store_data)):
    print(i)
    records.append([str(store_data.values[i,j]) for j in range(0, 20) if str(store_data.values[i,j]) != 'nan'])

records


from efficient_apriori import apriori


itemsets, rules = apriori(records,min_support=0.001)

rules

for rule in rules:
    print(rule)
    
    
    

rules_1 = list(filter(lambda rule: len(rule.lhs) <= 2 and len(rule.rhs) <= 2, rules))


#rules_1

for rule1 in rules_1:
    print (rule1)





from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


te = TransactionEncoder()
te_ary = te.fit(records).transform(records)
te_ary

df = pd.DataFrame(te_ary, columns=te.columns_)
df


itemsets = apriori(df, use_colnames=True, min_support=0.01)
itemsets
pd.set_option('display.max_columns',None)

rules = association_rules(itemsets,min_threshold=0.000001)
rdf = rules[['antecedents', 'consequents','support','confidence','lift']]

rdf
rdf.to_csv('rdf.csv')



# Case Sore Data




import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

#df = pd.read_excel('http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx')
df = pd.read_csv('29_Apriori/online_store.csv')
df.head()

df['Description'] = df['Description'].str.strip()
df
df['Description'].head(5)



df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)

df.dtypes

df['InvoiceNo'] = df['InvoiceNo'].astype('str')


df = df[~df['InvoiceNo'].str.contains('C')]





basket = (df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))
del df
basket.to_csv('basket.csv')


def encode_units(x):
    if x <= 0:
        return 0
    else:
        return 1

basket_sets = basket.applymap(encode_units)

basket_sets.drop('POSTAGE', inplace=True, axis=1)

frequent_itemsets = apriori(basket_sets, min_support=0.03, use_colnames=True)

frequent_itemsets

frequent_itemsets.to_csv("Freq_data.csv")

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.columns
rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
rules


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

rules

rules[rules['confidence']]

rdf [rdf ['confidence']]



#File handling


f = open('aa.txt')

f = open('aa.txt', mode='w')
f.write("Hello")
f.close()


f = open('aa.txt', mode='a')
f.write("Hello Python")
f.close()


f = open('aa.txt', mode='w')
f.write("Hello")
f.close()


f = open('aa.txt', mode='a')
f.write("Hello\n")
f.write("Hello Python Programming\n")
f.write("Analytics\n")
f.write("Data Science\n")
f.close()


f = open('aa.txt', mode='r')
f.read()
f.close()


f = open('aa.txt', mode='r')
f.readline()
f.readline()
f.readline()
f.close()


f = open('aa.txt', mode='r')

while(True):
    txt=f.readline()
    if(txt):
        print(txt)
    else:
        print("End of File")
        break
f.close()


def csvreader(file):
    f = open(file, mode='r')
    cols = f.readline().replace('\n','').split(',')
    print(cols)
    while(True):
        txt=f.readline()
        if(txt):
            print(txt.replace('\n','').split(','))
        else:
            
            break
    f.close()

    

csvreader('dat.csv')

csvreader('Ames_label.csv')



#Text Sentiment analysis


file = open('31_TextSentiment/metamorphosis.txt','r')
text = file.read()
file.close()
text



import re

def cleanString(text):
    res = []
    for i in text.strip().split():
        if not re.search(r"(https?)", i):   #Removes URL..Note: Works only if http or https in string.
            res.append(re.sub(r"[^A-Za-z\.]", "", i).replace(".", " "))   #Strip everything that is not alphabet(Upper or Lower)
    return " ".join(map(str.strip, res))


text = cleanString(text)


text






file = open('31_TextSentiment/metamorphosis.txt','r')
text = file.read()
file.close()
text

text =text.replace('\n',' ')
text

import string
for i in string.punctuation:
    text=text.replace(i,'')

text
text = cleanString(text)
text

words = text.split()
words


import nltk
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

stemmed = [porter.stem(word) for word in words]


import nltk

nltk.download('vader_lexicon')
nltk.download('punkt')



import re
words_re = re.compile(" ".join(words))
word_str = str(words_re)
word_str


t1 = "This is good"
t2 = "This is best"
t3 = "This is average"
t4 = 'This is bad'
t5 = "This is worst"

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
sid.polarity_scores(t1)

sid.polarity_scores(t2)
sid.polarity_scores(t3)
sid.polarity_scores(t4)
sid.polarity_scores(t5)


t6 = "It's have been a great learning experience in the Spanish Language learning. I was enrolled in the Spanish Language A1 course from Henry Harvin and currently, I am preparing for my exams so that I can apply for the desired job in Spain. The faculty not only helped me to clear my doubts but strengthened my confidence to speak Spanish fluently. Though the price is a bit high-priced, it is worthwhile.."

def cleanString(text):
    res = []
    for i in text.strip().split():
        if not re.search(r"(https?)", i):   #Removes URL..Note: Works only if http or https in string.
            res.append(re.sub(r"[^A-Za-z\.]", "", i).replace(".", " "))   #Strip everything that is not alphabet(Upper or Lower)
    return " ".join(map(str.strip, res))


text = cleanString(t6)


sid.polarity_scores(text)


t7 = "Henry Harvin truly has changed my life!!! It is an amazing institute for learning the Spanish language, having an experienced and encouraging faculty. The teachers make everything seem easy and fun and there is always a friendly atmosphere in the class. When I started learning Spanish, I had never thought I would come so far, this could become possible under the guidance of Pankaj Sir and other teachers. I truly believe that at Henry Harvin anyone can learn anything!!!! Plus with lots of fun. The best institution for languages.."
text = cleanString(t7)

sid.polarity_scores(text)



t8 = "Don't trust fake reviews most of the reviews are posted by it's own employees. As they are forcing them to post good reviews. Henry Harvin has nothing better than other academies. And it's services are very poor. If you give money to Hanry Harvin get ready for no results. And delay in classes or maybe no classes. They will take your money and then you'll be ignored. And never ever work with Hanry Harvin otherwise no guarantee for your growth. I can give you the details of it's employees so that you can discuss more about Henry Harvin."

text = cleanString(t8)

sid.polarity_scores(text)



t9 = "This is not bad"

text = cleanString(t9)

sid.polarity_scores(text)


t9 = "I thought it was worst but it was other way around"

text = cleanString(t9)

sid.polarity_scores(text)



#News Case Study



from newsapi import NewsApiClient

# Init
newsapi = NewsApiClient(api_key='ec4604b2a608472dbf57f26b938ed8b9')

# /v2/top-headlines
top_headlines = newsapi.get_top_headlines(q='bitcoin',
                                          sources='bbc-news,the-verge',
                                          )

top_headlines
top_headlines['status']
type(top_headlines)
top_headlines.keys()

top_headlines['articles']

type(top_headlines['articles'])

len(top_headlines['articles'])

top_headlines['articles'][0]

type(top_headlines['articles'][0])

top_headlines['articles'][0].keys()

top_headlines['articles'][0]['title']
text = top_headlines['articles'][0]['description']

def cleanString(text):
    res = []
    for i in text.strip().split():
        if not re.search(r"(https?)", i):   #Removes URL..Note: Works only if http or https in string.
            res.append(re.sub(r"[^A-Za-z\.]", "", i).replace(".", " "))   #Strip everything that is not alphabet(Upper or Lower)
    return " ".join(map(str.strip, res))


text = cleanString(text)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
sid.polarity_scores(text)
text




#News Case Study 1

from newsapi import NewsApiClient

# Init
newsapi = NewsApiClient(api_key='ec4604b2a608472dbf57f26b938ed8b9')

# /v2/top-headlines
top_headlines = newsapi.get_top_headlines(q='russia')

def cleanString(text):
    res = []
    for i in text.strip().split():
        if not re.search(r"(https?)", i):   #Removes URL..Note: Works only if http or https in string.
            res.append(re.sub(r"[^A-Za-z\.]", "", i).replace(".", " "))   #Strip everything that is not alphabet(Upper or Lower)
    return " ".join(map(str.strip, res))


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()


pol=[]
txt=[]
for i in range(len(top_headlines['articles'])):
    text = top_headlines['articles'][i]['description']
    text = cleanString(text)
    txt.append(text)
    pol.append(str(sid.polarity_scores(text)['compound']))


import pandas as pd
pd.DataFrame({'Text':txt, 'Polarity':pol}).to_csv("NewsAnalysis.csv")



#News Case Study 2

from newsapi import NewsApiClient

# Init
newsapi = NewsApiClient(api_key='ec4604b2a608472dbf57f26b938ed8b9')

# /v2/top-headlines
top_headlines = newsapi.get_top_headlines(q='Market')

def cleanString(text):
    res = []
    for i in text.strip().split():
        if not re.search(r"(https?)", i):   #Removes URL..Note: Works only if http or https in string.
            res.append(re.sub(r"[^A-Za-z\.]", "", i).replace(".", " "))   #Strip everything that is not alphabet(Upper or Lower)
    return " ".join(map(str.strip, res))


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()


pol=[]
txt=[]
for i in range(len(top_headlines['articles'])):
    text = top_headlines['articles'][i]['description']
    text = cleanString(text)
    txt.append(text)
    pol.append(str(sid.polarity_scores(text)['compound']))

txt

import pandas as pd
pd.DataFrame({'Text':txt, 'Polarity':pol}).to_csv("NewsAnalysis.csv")




# tweepy


import tweepy  #pip install tweepy
import csv
import pandas as pd

#credentials
'''
https://dev.twitter.com/apps/new
'''

APIKey='Duab8EZMe9KrdLXtRDjHHf5EU'
APISecret='ungnDR16pqraP0LT6bsGI2OqKOC1CY3tenwRO2ETyqQsDSo7kq'
AccessToken='144501392-rA4XGtpVuDccWXeuZhZMO2rtA53ow7FtK8VGdEii'
AccessTokenSecret='oq8bQCMtrxt8tiofJ7hMM6UjwAddMLMvUBleSRYaUihYi'



auth = tweepy.OAuthHandler(APIKey, APISecret)
auth.set_access_token(AccessToken, AccessTokenSecret)
api = tweepy.API(auth, wait_on_rate_limit=True)



handle =['Reliance Industries']

import pandas as pd

creat = []
txt = []

for tweets in api.search_tweets(q=handle, count =100, lang="en"):
    print(tweets.created_at, tweets.text.encode('utf-8'))
    creat.append(tweets.created_at)
    txt.append(tweets.text)



import re
def cleanString(text):
    res = []
    for i in text.strip().split():
        if not re.search(r"(https?)", i) and not re.search(r"(@)", i):   #Removes URL..Note: Works only if http or https in string.
            res.append(re.sub(r"[^A-Za-z\.]", "", i).replace(".", " "))   #Strip everything that is not alphabet(Upper or Lower)
        
    return " ".join(map(str.strip, res))


text=[]
for i in txt:
    text.append(cleanString(i))
    
text



df = pd.DataFrame({'creat':creat, 'txt':text})
df.to_csv("tweet.csv")




import string
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()


scr = []

for i in range(0, len(df)):
    tx = df.iloc[i]['txt']
    for i in string.punctuation:
        tx = tx.replace(i,'')
    scores = sid.polarity_scores(tx)
    scr.append(scores['compound'])
    

df['score'] = scr


df.to_csv('tweetscore.csv')




























































































    
    
    
    




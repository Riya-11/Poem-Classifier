import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re
import string

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import nltk 
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.corpus import stopwords

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.metrics import classification_report,confusion_matrix

df = pd.read_csv('dataset.csv')
print("\nDataset Description:\n ")

df.info()
print("----------------------------\n\n")
print("Dataset Categories\n")
print(df.groupby('type').count())
print("----------------------------\n\n")

def removePunctuation(x): #function to remove punctuation
    x = x.lower()
    x = re.sub(r'[^\x00-\x7f]',r' ',x)
    x = x.replace('\r','')
    x = x.replace('\n','')
    x = x.replace('  ','')
    x = x.replace('\'','')
    return re.sub("["+string.punctuation+"]", " ", x)

stops = set(stopwords.words("english")) 
def removeStopwords(x): #function to remove stopwords
    filtered_words = [word for word in x.split() if word not in stops]
    return " ".join(filtered_words)

def processText(x):
    x= removePunctuation(x)
    x= removeStopwords(x)
    return x

X_train, X_test, y_train, y_test = train_test_split(df['content'],df['type'],test_size = 0.2,random_state = 96 )

pipeline1 = Pipeline([
        ('bow', CountVectorizer()),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('c',LogisticRegression())
    ])

clf1 = pipeline1.fit(X_train,y_train)

pred=pipeline1.predict(X_test)

print("Classification results using Logistic Regression as the classifier\n")
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
print("Accuracy : ",accuracy_score(pred,y_test))
print("----------------------------\n\n")


pipeline2 = Pipeline([
        ('bow', CountVectorizer()),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('c',LinearSVC()),
    ])

clf2 = pipeline2.fit(X_train,y_train)

pred=pipeline2.predict(X_test)
print("Classification results using LinearSVC as the classifier\n")

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
print("Accuracy : ",accuracy_score(pred,y_test))
print("----------------------------\n")
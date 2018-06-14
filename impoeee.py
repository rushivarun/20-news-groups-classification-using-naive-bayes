# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 20:56:34 2018

@author: Rushi Varun
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from sklearn.pipeline import Pipeline

stuff_to_train = fetch_20newsgroups(subset = "train", shuffle = True)
stuff_to_train.target.shape 
stuff_to_test = fetch_20newsgroups(subset='test', shuffle=True)
text_clf = Pipeline([('vect',CountVectorizer()),('clf',MultinomialNB())])
text_clf = text_clf.fit(stuff_to_test.data, stuff_to_test.target)
predicted = text_clf.predict(stuff_to_test.data)
print(np.mean(predicted == stuff_to_test.target))
stuff_to_train.target.shape
print(classification_report(stuff_to_test.target,predicted))
text = input("enter the text you want to categorise") # custom input
predict_new = text_clf.predict([text])
targetNames = stuff_to_train.target_names

print(targetNames[int(predict_new)])
mydict = {"text" : targetNames[int(predict_new)]}
print(mydict)


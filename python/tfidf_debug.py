
# import dependencies
import csv as csv
import numpy as np
import pandas as pd
import pylab as py
import operator
import re, os, progressbar
import sys
import nltk
from pprint import *
from sklearn.manifold import TSNE
import multiprocessing
import codecs
import gensim.models.word2vec as w2v
from nltk.corpus import stopwords
from nltk import FreqDist
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as skmetrics
from sklearn.metrics import confusion_matrix, f1_score
#import plot_tools as myplt
from sklearn.externals import joblib
from operator import itemgetter
import pickle
import logging

fake_data = pd.read_csv('data/fake.csv')
fake_data = fake_data[fake_data.language == 'english']
fake_data.dropna(axis=0, inplace=True, subset=['text'])
#fake_data = fake_data.sample(frac=1.0)
fake_data.reset_index(drop=True,inplace=True)
fake_data.describe()

# Now the Real Data
real_data = pd.read_csv('data/real_news.csv')
real_data = real_data[fake_data.language == 'english']
real_data.dropna(axis=0, inplace=True, subset=['text'])
#real_data = real_data.sample(frac=1.0)
real_data.reset_index(drop=True,inplace=True)

# Add category names(Fake, Real) to their respective dataset
fake_data['class'] = 0
real_data['class'] = 1

# Combine the real and fake for training(to be used later on).  We will randomize the fake and real sets together.
fake_and_real_data = pd.concat([fake_data, real_data]).sample(frac=1.0)

# Vectorize the full text and check all dimensions
cv = CountVectorizer(stop_words='english')
tfidf_transformer = TfidfTransformer()

# PreTransoform dimensions
print('\nDF Dimensions with CLass labels (Rows = articles, Columns = non vec features):\n', fake_and_real_data.shape)

# CV step.  Sci kit makes scipy sparse matrices.  Conver to numoy array
cv_text = cv.fit_transform(fake_and_real_data['text']).toarray()

# What dimensions do we have now
print('\nDF Dimensions of CV (Rows = articles, Columns = words):\n', cv_text.shape)
print('Type of Sci-Kit CV Array:', type(cv_text))

# append labels to vectorized word numpy array
label_array = np.asarray(fake_and_real_data['class'].values.reshape(fake_and_real_data['class'].shape[0], 1))
print('\nClass Label Array:\n', label_array.shape) 

test = np.append(cv_text, label_array, axis=1)

# What dimensions do we have now
print('\nDF Dimensions of CV (Rows = articles, Columns = words + 1 label):\n', test.shape)

# How many real and fake
print('\n# Fake in CV:', test[test[:,-1] == 0].shape)
print('# Real in CV:', test[test[:,-1] == 1].shape)

# tfidf step
tfidf_text =  tfidf_transformer.fit_transform(cv_text)

# What dimensions do we have now
#print('\nDF Dimensions of TFIDF (Rows = articles, Columns = words):\n', tfidf_text.shape)

#tfidf_test = np.append(tfidf_text, label_array, axis=1)

# What dimensions do we have now
#print('\nDF Dimensions of TFIDF (Rows = articles, Columns = words + 1 label):\n', tfidf_test.shape)

# How many real and fake
#print('\n# Fake in TFIDF:', tfidf_test[test[:,-1] == 0].shape)
#print('# Real in TFIDF:', tfidf_test[test[:,-1] == 1].shape)

# Train the TFIDF


if not os.path.exists('tfidf_test.p'):
    tfidf_mnb = MultinomialNB() 
    tfidf_mnb.fit(tfidf_text, fake_and_real_data['class'])
    pickle.dump(tfidf_mnb, open('tfidf_test.p', 'wb')) 
else:
    tfidf_mnb = pickle.load(open('tfidf_test.p', 'rb'))
    
predictions = tfidf_mnb.predict(tfidf_text)

predictions = predictions.reshape(predictions.shape[0], 1) 

print('\nPredictions Type:', type(predictions))

print('\n# Fake in Predictions:', predictions[predictions[:] == 0].shape)
print('# Real in Predictions:', predictions[predictions[:] == 1].shape)

#print(confusion_matrix(fake_and_real_data['class'].values, predictions))
#print(confusion_matrix(predictions, fake_and_real_data['class'].values))

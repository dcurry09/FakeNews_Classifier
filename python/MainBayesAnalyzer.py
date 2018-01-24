##########################################
#  MainBayesAnalyzer.py
#
# Wrapper for analyzing real/fake datasets
# Uses Bayes Classifier in Sci-Kit
#
# By David Curry,  Oct 2017
#
##########################################

import csv as csv
import numpy as np
import pandas as pd
import pylab as py
import operator
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# From Kaggle
fake_data = pd.read_csv('data/fake.csv', nrows=100)
fake_data = fake_data[fake_data.language == 'english']

# Tell me a bit about the initial form of the fake data
#print ('\nFake Data Structure:', fake_data.head(5))
#print (fake_data.describe())
#print ('Initial Dimensions(rows, columns):', fake_data.shape)

print ('Making the Count Vector...')

count_vect = CountVectorizer()
fake_matrix_CV = count_vect.fit_transform(fake_data['text'].values.astype('U'))
fake_features  = count_vect.get_feature_names()

# Vocab dict is each name with its unique ID, not its frequency
fake_vocab = count_vect.vocabulary_

# Sort the vocab by frequency
vocab_sorted = [v[0] for v in sorted(fake_vocab.items(), key=operator.itemgetter(1))]

fdist_fake_vocab = freq_distribution = Counter(dict(zip(vocab_sorted, np.ravel(fake_matrix_CV.sum(axis=0)))))

print ('Sorted Vocab List:', fdist_fake_vocab)

#plt.bar(fake_vocab_sorted_list[:10], di, color='g')
#plt.show()








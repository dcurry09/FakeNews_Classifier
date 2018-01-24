# Word2Vec
# Up to this point we had considered our data in word "chunks", that is we broke each sentence into individual parts. In doing so we may have lost a bit of descriptive due to ignoring the holistic nature of words within the document from which it came. Now we will look at words in the context of the sentences in which they reside. To do so we will use the NLTK package for cleaning and tokenization by sentence.
# Once sentences have been tokenized we will use Word2vec to cluster words together and to find patterns within sentences and documents. Developed at Google, Word2Vec is an unsupervised 2-layer neural network that uses backpropagation to assign probabilites for each word to occur near another for in a given interval. The hyperparameters of the NN are:
# num_features is the number of dimensions we want to embed our words in (the length of the final word vectors)
# min_word_count, if the count of a words use is below this threshold, corpus wide, ignore the word
# context_size is the window size left and right of the target word to consider words

# import dependencies
import csv as csv
import numpy as np
import pandas as pd
import pylab as py
import operator
import re, pickle
import sys, os
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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as skmetrics
from sklearn.metrics import confusion_matrix, f1_score
import plot_tools as myplt
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
count_vect = CountVectorizer(stop_words="english")
realfake_matrix_CV = count_vect.fit_transform(fake_and_real_data['text'].values.astype('U'))
features = count_vect.get_feature_names()


# Train the data
CV_pipeline_optimized = Pipeline([
    ('CountVectorizer',  CountVectorizer(stop_words="english", ngram_range=(1, 5))),
    ('MNBclassifier',  MultinomialNB(alpha=0.01, fit_prior=True))])



# Now make the word2vec embeddings
def sent_token(text):
    text = nltk.sent_tokenize(text)
    return text

def sentence_clean(text):
    new_text = []
    for sentence in text:
        sentence = sentence.lower()
        sentence = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", sentence)
        sentence = re.sub("[^a-z ]", "", sentence)
        sentence = nltk.word_tokenize(sentence)
        sentence = [word for word in sentence if len(word)>1] # exclude 1 letter words
        new_text.append(sentence)
    return new_text

def apply_all(text):
    return sentence_clean(sent_token(text))

def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.
    '''

    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'CV']
    return df

def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    ''' 
    Top tfidf features in specific document (matrix row) 
    '''
    
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. 
        Xtr = tfidf array
        features = names from vocab(list)
    '''
    
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)



def kfold_validate(data, pipeline):
    ''' Performs kFold Cross-Validation on text classes.
        Inputs: (pd.dataset, scikit.pipeline)
        Returns: [title, f1_scores, confusion matrix, class_report]
    '''
    
    title = [str(step) for step in pipeline.named_steps]
    title = '_'.join(title)
    print('\nK-Fold Validation on Model:', title)
    
    k_fold = KFold(n=len(data), n_folds=6)
    scores = []
    y_true = []
    y_pred = [] 
    confusion = np.array([[0, 0], [0, 0]])
    
    # Keep track of pass/fail events
    false_pos = set()
    false_neg = set()
    true_pos  = set() 
    true_neg  = set() 
        

    for train_indices, test_indices in k_fold:
        #train_text = data.iloc[train_indices]['text'].values
        #train_y = data.iloc[train_indices]['class'].values

        #test_text = data.iloc[test_indices]['text'].values
        #test_y = data.iloc[test_indices]['class'].values

        train_text = data.iloc[train_indices]['text']
        train_y = data.iloc[train_indices]['class']

        test_text = data.iloc[test_indices]['text']
        test_y = data.iloc[test_indices]['class']
        
        print('\nFitting Fold...')
        
        pipeline.fit(train_text, train_y)
        
        print('Making Predictions...')
        
        predictions = pipeline.predict(test_text)
        
        confusion += confusion_matrix(test_y, predictions)
        score = f1_score(test_y, predictions, pos_label=0)
        scores.append(score)
        y_true.extend(test_y)
        y_pred.extend(predictions)    
        
        # Collect indices of false positive and negatives
        test_predict = np.squeeze(pipeline.predict(test_text))
        fp_i = np.where((test_predict==1) & (test_y==0))[0]
        fn_i = np.where((test_predict==0) & (test_y==1))[0]
        tn_i = np.where((test_predict==0) & (test_y==0))[0]
        tp_i = np.where((test_predict==1) & (test_y==1))[0]
        false_pos.update(test_indices[fp_i])
        false_neg.update(test_indices[fn_i])
        true_pos.update(test_indices[tp_i])
        true_neg.update(test_indices[tn_i])
        
        
        
    tp, fp, fn, tn = confusion.flatten()
    measures = {}
    measures['accuracy'] = (tp + tn) / (tp + fp + fn + tn)
    measures['specificity'] = tn / (tn + fp)        # (true negative rate)
    measures['sensitivity'] = tp / (tp + fn)        # (recall, true positive rate)
    measures['precision'] = tp / (tp + fp)
    measures['f1score'] = 2*tp / (2*tp + fp + fn)    
        
    print('News Articles classified:', len(data))
    print('Score:', sum(scores)/len(scores))
    print('Confusion matrix:')
    print(confusion)
    pprint(measures)

    return [title, sum(scores)/len(scores), confusion, measures, y_true, y_pred, 
            {'fp': sorted(false_pos), 'fn': sorted(false_neg), 'tp': sorted(true_pos), 'tn': sorted(true_neg)}]
    

if not os.path.exists('cv_results_optmized.p'):
    cv_results_optmized = kfold_validate(fake_and_real_data, CV_pipeline_optimized)
    pickle.dump(cv_results_optmized, open('cv_results_optmized.p', 'wb')) 
else:
    cv_results_optmized = pickle.load(open('cv_results_optmized.p', 'rb'))
    print('\nLoaded Optimized MNB Results:\n', cv_results_optmized[0], cv_results_optmized[2], cv_results_optmized[3])


fake_and_real_data['sent_tokenized_text'] = fake_and_real_data['text'].apply(apply_all)
all_sentences = list(fake_and_real_data['sent_tokenized_text'])
all_sentences = [subitem for item in all_sentences for subitem in item]

num_features = 300 
min_word_count = 10
num_workers = multiprocessing.cpu_count()
context_size = 7 
downsampling = 1e-3 
fake2vec = w2v.Word2Vec(
    sg=1,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)
fake2vec.corpus_count
fake2vec.build_vocab(all_sentences)


if not os.path.exists('w2v_matrix.p'):

    print("Word2Vec vocabulary length:", len(fake2vec.wv.vocab))
    print('\nTraining Word2Vec...')
    fake2vec.train(all_sentences, total_examples=fake2vec.corpus_count, epochs=fake2vec.iter)
    all_word_vectors_matrix = fake2vec.wv.syn0
    pickle.dump(all_word_vectors_matrix, open('w2v_matrix.p', 'wb')) 
    pickle.dump(fake2vec, open('w2v_fit.p', 'wb')) 
else:    
    all_word_vectors_matrix = pickle.load(open('w2v_matrix.p', 'rb'))
    print('\nLoaded Word2Vec Results...')


#Visualizing Word2Vec output
#A common technique to visualize our newly clustered words is to use TSNE(https://lvdmaaten.github.io/tsne/). TSNE utilizes dimensionality reduction in order to visualize our embedded word relationships in a 2-D representation.
#Lets look at clusters of all words first, then top 1000, then top 10.

#Each cluster in this plot can be thought of as a new feature that could be used for training in another algroithm. Instead of hundreds of thousands of unique word features, word2Vec has provided us with a reduced feature. Whether or not this reduced feature set will provide better seperation power versus real news features will be one of the questions I will look at as this project continues. Below are the first 1000 and 10 words from Word2Vec training and a 2-D projection from TSNE.

if not os.path.exists('tsne_fit.p'):
    tsne = TSNE(n_components=2, random_state=0)
    #np.set_printoptions(suppress=True)
    Y_all = tsne.fit_transform(all_word_vectors_matrix)
    pickle.dump(Y_all, open('tsne_fit.p', 'wb')) 
else:
    Y_all = pickle.load(open('tsne_fit.p', 'rb'))
    print('\nLoaded TSNE Fit Results')    


plt.figure(figsize=(20,12))
plt.scatter(Y_all[:, 0], Y_all[:, 1])
#plt.savefig('_w2v_tsne.pdf')
#plt.show()


# Now lets highlight common words to the False Negative category
df_CV_FN_100 = top_mean_feats(realfake_matrix_CV, features, grp_ids=cv_results_optmized[6]['fn'], top_n=500)
#df_CV_FN_100 = top_mean_feats(realfake_matrix_CV, features, grp_ids=cv_results_optmized[6]['fn'])

plt.figure(figsize=(20,12))

xy_fn, xy = [],[]

#xy_fn = [x,y for ]

print(fake2vec.wv.vocab[1])

for label, x, y in zip(fake2vec.wv.vocab, Y_all[:, 0], Y_all[:, 1]):
    
    #print('Label:', label, ' feature:', str(label) in df_CV_FN_100['feature'])
        
    if str(label) in df_CV_FN_100['feature']:
        print("FN Feature:", label)
        plt.scatter(x,y, color='r', s=10)
        xy_fn.append(x,y) 
    else: 
        plt.scatter(x,y, s=10)
        #xy.append(x,y)
    
    #plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

plt.title('TSNE Map of Word2Vec Embeddings: FN')
plt.savefig('all_FN_tsne.pdf')
plt.show()



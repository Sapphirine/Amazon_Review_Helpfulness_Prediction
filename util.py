from collections import namedtuple
from itertools import izip
from multiprocessing import Pool
from nltk import PorterStemmer
from nltk import SnowballStemmer
from nltk import word_tokenize
from scipy.sparse import hstack
from scipy.sparse.csr import csr_matrix
from sklearn import cross_validation
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import RFECV
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost.sklearn import XGBClassifier
import cPickle
import datetime
import gensim
import itertools
import json
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
import re
import scipy
import sklearn
import sklearn.ensemble as ensemble
import sklearn.feature_extraction as feature_extraction
import sklearn.metrics as metrics
import sklearn.naive_bayes as naive_bayes
import sklearn.preprocessing as preprocessing
import time
import xgboost as xgb

class Tokenizer:
    def __init__(self, stemming=False):
        if stemming:
            self.stemmer = PorterStemmer().stem
        else:
            self.stemmer = None
        self.tokenize = word_tokenize
    def __call__(self, s):
        if self.stemmer == None:
            return self.tokenize(s)
        else:
            return map(self.stemmer, self.tokenize(s))

class FastTokenizer:
    def __init__(self, stemming=False):
        if stemming:
            self.stemmer = SnowballStemmer('english').stem
        else:
            self.stemmer = None
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
        self.tokenize = token_pattern.findall
    def __call__(self, s):
        if self.stemmer == None:
            return self.tokenize(s)
        else:
            return map(self.stemmer, self.tokenize(s))

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r %2.2f sec' % \
              (method.__name__, te-ts)
        return result
    return timed

@timeit
def read_review_json_file(filename, meta_filename, take=lambda x:True):
    meta = dict()
    with open(meta_filename, 'r') as input_file:
        for line in input_file:
            # this is eval because it is not a standard json
            meta_content = eval(line)
            meta[meta_content['asin']] = meta_content
    
    Row = namedtuple('Row', ['ups', 'totals', 'overall', 'review', 'summary', 'reviewer_id', 'reviewer_name',
                            'review_time', 'product_id', 'brand', 'categories', 'price', 'product_title',
                            'review_year', 'review_month'])
    rows = []
    
    line_count = 0
    max_line_count = None
    with open(filename, 'r') as input_file:
        for line in input_file:
            review = json.loads(line)
            totals = review['helpful'][1]
            ups = review['helpful'][0]
            overall = float(review['overall'])
            review_text = review['reviewText']
            summary = review['summary']
            reviewer_id = review['reviewerID']
            reviewer_name = review.get('reviewerName', 'Unknown')
            review_time = int(review['unixReviewTime'])
            product_id = review['asin']
            # for product level info
            brand = meta[product_id].get('brand', 'Unknown')
            categories = meta[product_id].get('categories', [['Unknown']])
            price = float(meta[product_id].get('price', -999))
            product_title = meta[product_id].get('title', 'Unknown')
            dt = datetime.datetime.fromtimestamp(review_time)
            review_year = dt.year
            review_month = dt.month
            row = Row(ups=ups, totals=totals, overall=overall, review=review_text, summary=summary,
                            reviewer_id=reviewer_id, reviewer_name=reviewer_name,review_time=review_time,
                            product_id=product_id, brand=brand, categories=categories, price=price,
                            product_title=product_title, review_year=review_year, review_month=review_month)
            if take(row):
                rows.append(row)
            line_count += 1
            if max_line_count is not None and line_count >= max_line_count:
                break
    df = pd.DataFrame(rows, columns=Row._fields)
    return df
def get_train_test_split(n, train_ratio=0.7):
    rand_perm = np.random.permutation(n)
    train_indexes = rand_perm[:int(n * train_ratio)]
    test_indexes = rand_perm[int(n * train_ratio):]
    return train_indexes, test_indexes
def populate_fields(df):
    df['ratio'] = (df['ups']) / (df['totals'] + 0.0001) 
    df['downs'] = df['totals'] - df['ups']
    df['has_ups'] = df['ratio'] > 0
    df['has_votes'] = df['totals'] > 0
    df['length'] = map(lambda x: len(x.split()), df['review'])

    group = df[['product_id', 'review_time']].groupby(["product_id"], as_index=False).min()
    first_review_dict = dict(np.asarray(group[['product_id', 'review_time']]))
    df['review_lateness'] = map(lambda review_time, product_id: review_time - first_review_dict[product_id],
                                  df['review_time'], df['product_id'])

    count_group = df[['product_id']].groupby(["product_id"], as_index=False).size()
    review_number_dict = dict(zip(count_group.index, count_group.values))
    df['product_review_number'] = map(lambda product_id: review_number_dict[product_id], df['product_id'])
    return df

def show_feature_stats(name_to_feature_dict):
    for name in name_to_feature_dict:
        if name_to_feature_dict[name].shape:
            try:
                print name, '[', name_to_feature_dict[name].min(), ',', name_to_feature_dict[name].max(), ']', 'shape:',  name_to_feature_dict[name].shape
            except:
                pass

def evaluate(true_y, pred_y_prob, verbose=False, draw_roc=False, draw_precision_recall_curve=False):
    print 'auc: ', metrics.average_precision_score(true_y, pred_y_prob)
    best_f1 = -1
    best_threshold = None
    for threshold in np.linspace(0.01, 1, 100):
        pred_y = pred_y_prob >= threshold
        f1 = f1_score(true_y, pred_y)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
        if verbose:
            print 'threshold: ', threshold
            print 'accuracy: ', metrics.accuracy_score(true_y, pred_y)
            print 'precision: ', metrics.precision_score(true_y, pred_y)
            print 'recall: ', metrics.recall_score(true_y, pred_y)
            print 'f1: ', f1_score(true_y, pred_y)
    pred_y = pred_y_prob >= best_threshold
    print 'threshold: ', best_threshold
    print 'accuracy: ', metrics.accuracy_score(true_y, pred_y)
    print 'precision: ', metrics.precision_score(true_y, pred_y)
    print 'recall: ', metrics.recall_score(true_y, pred_y)
    print 'f1: ', f1_score(true_y, pred_y)
    if draw_roc:
        fprs, tprs, _ = roc_curve(true_y, pred_y_prob)
        roc_auc = auc(fprs, tprs)
        plt.figure()
        plt.plot(fprs, tprs, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()
    if draw_precision_recall_curve:
        precisions, recalls, _ = precision_recall_curve(true_y, pred_y_prob)
        average_precision = average_precision_score(true_y, pred_y_prob)
        plt.clf()
        plt.plot(recalls, precisions, label='Precision-Recall curve (area = %.2f)'%average_precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend(loc="lower left")
        plt.show()

def get_features_by_names(name_to_feature_dict, used_feature_names, scale_feature_names={}):
    candidates = []
    for name in used_feature_names:
        if name in scale_feature_names:
            candidates.append(preprocessing.scale(name_to_feature_dict[name].todense()))
        else:
            candidates.append(name_to_feature_dict[name])
    return hstack(candidates, format='csr')

def get_data_split(size, seed=42):
    np.random.seed(42)
    # split data set into dev set and test set
    dev_indexes, test_indexes = get_train_test_split(size, train_ratio=0.7)
    # And split dev set into training set and validation set
    train_indexes_raw, valid_indexes_raw = get_train_test_split(len(dev_indexes), train_ratio=0.7)
    train_indexes = dev_indexes[train_indexes_raw]
    valid_indexes = dev_indexes[valid_indexes_raw]
    return train_indexes, valid_indexes, test_indexes
def extract_features(data, train_indexes, do_not_consider=set()): 
    review_vectorizer = feature_extraction.text.TfidfVectorizer(max_features=20000, min_df=5, max_df=0.5, lowercase=True, tokenizer=FastTokenizer(stemming=True))
    review_vectorizer.fit(data['review'][train_indexes])
    vectorized_reviews = review_vectorizer.transform(data['review'])

    if 'summary' not in do_not_consider:
        summary_vectorizer = feature_extraction.text.TfidfVectorizer(min_df=5, max_df=0.5, lowercase=True, tokenizer=FastTokenizer(stemming=True))
        summary_vectorizer.fit(data['summary'][train_indexes])
        vectorized_summary = summary_vectorizer.transform(data['summary'])
    else:
        vectorized_summary = []

    reviwer_id_vectorizer = feature_extraction.text.HashingVectorizer(n_features=20000,lowercase=False, tokenizer=lambda x:[x], non_negative=True)
    reviwer_id_vectorizer.fit(data['reviewer_id'][train_indexes])
    vectorized_reviewer_id = reviwer_id_vectorizer.transform(data['reviewer_id'])

    reviewer_name_vectorizer = feature_extraction.text.HashingVectorizer(n_features=5000, lowercase=True, tokenizer=lambda x:x.split(), non_negative=True)
    reviewer_name_vectorizer.fit(data['reviewer_name'][train_indexes])
    vectorized_reviewer_name = reviewer_name_vectorizer.transform(data['reviewer_name'])

    product_id_vectorizer = feature_extraction.text.HashingVectorizer(n_features=20000,lowercase=False, tokenizer=lambda x:[x], non_negative=True)
    product_id_vectorizer.fit(data['product_id'][train_indexes])
    vectorized_product_id = product_id_vectorizer.transform(data['product_id'])

    category_vectorizer = feature_extraction.text.HashingVectorizer(n_features=5000, lowercase=False, tokenizer=lambda x:x[0], non_negative=True)
    category_vectorizer.fit(data['categories'][train_indexes])
    vectorized_category = category_vectorizer.transform(data['categories'])

    review_title_vectorizer = feature_extraction.text.HashingVectorizer(n_features=20000, lowercase=False, tokenizer=lambda x:x[0], non_negative=True)
    review_title_vectorizer.fit(data['product_title'][train_indexes])
    vectorized_product_title = review_title_vectorizer.transform(data['product_title'])

    if 'word_embedding' not in do_not_consider:
        tokenizer = FastTokenizer()
        get_iter = lambda: itertools.imap(lambda x: tokenizer(x.lower()), data['review'][train_indexes])
        word2vec_model = gensim.models.Word2Vec(min_count=10, workers=8)
        word2vec_model.build_vocab(get_iter())
        word2vec_model.train(get_iter())
        mean_vecs = list()
        for tokenized_doc in itertools.imap(lambda x: tokenizer(x.lower()), data['review']):
            mean_vec = np.zeros(word2vec_model.vector_size)
            count = 0
            for word in tokenized_doc:
                if word in word2vec_model:
                    mean_vec += word2vec_model[word]
                    count += 1
            if count != 0:
                mean_vec /= count
            mean_vecs.append(mean_vec)
    else:
        mean_vecs = []

    feature_name_pairs = ((vectorized_reviews, 'vectorized_reviews'),
                (scipy.sparse.csr.csr_matrix(mean_vecs), 'word_embedding'),
                (scipy.sparse.csr.csr_matrix(data[['length']]), 'length'),
                (scipy.sparse.csr.csr_matrix(data[['overall']]), 'overall'),
                (scipy.sparse.csr.csr_matrix(data[['price']]), 'price'),
                (vectorized_reviewer_name, 'vectorized_reviewer_name'),
                (vectorized_category, 'vectorized_category'),
                (vectorized_product_id, 'vectorized_product_id'),
                (scipy.sparse.csr.csr_matrix(data[['review_time']]), 'review_time'),
                (scipy.sparse.csr.csr_matrix(data[['review_lateness']]), 'review_lateness'),
                (scipy.sparse.csr.csr_matrix(data[['review_year']]), 'review_year'),
                (vectorized_product_title, 'vectorized_product_title'),
                (scipy.sparse.csr.csr_matrix(data[['product_review_number']]), 'product_review_number'),
                )
    name_to_feature_dict = dict((x[1], x[0]) for x in feature_name_pairs)
    return name_to_feature_dict

def extract_leaf_feature(features, targets, train_indexes, params):
    model = XGBClassifier(**params)
    model.fit(features[train_indexes], targets[train_indexes])
    booster = model.booster()
    dmatrix = xgb.DMatrix(features)
    leaf = booster.predict(dmatrix, pred_leaf=True)
    encoder = sklearn.preprocessing.OneHotEncoder()
    leaf_feature = encoder.fit_transform(leaf)
    return leaf_feature

def train_and_evaluate(model, features, data, train_indexes, valid_indexes, test_indexes):
    model.fit(features[train_indexes], data.loc[train_indexes]['target'])

    print 'training set'
    train_prob_preds = model.predict_proba(features[train_indexes])
    evaluate(data.loc[train_indexes]['target'], train_prob_preds[:,1])

    print ''
    print 'validation set'
    valid_prob_preds = model.predict_proba(features[valid_indexes])
    evaluate(data.loc[valid_indexes]['target'], valid_prob_preds[:,1])

    print ''
    print 'test set'
    test_prob_preds = model.predict_proba(features[test_indexes])
    evaluate(data.loc[test_indexes]['target'], test_prob_preds[:,1])
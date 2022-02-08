#IMPORTS
import pandas as pd
import numpy as np
import regex as re
import string
from datetime import datetime
import nltk

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D
from keras import regularizers, Input
from keras.models import Sequential, Model
from keras.layers import Dense, concatenate, GlobalMaxPooling2D

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn import metrics

def preprocess_data(dataset_path, stops, stemmer):
    def preprocess_tweet(tweet, stops, stemmer):
        # remove URLs
        tweet = re.sub(r'http\S+', '', tweet)
        # remove users
        tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)
        # remove punctuation
        table = str.maketrans('', '', string.punctuation)
        tweet = tweet.translate(table)
        # remove stopwords & stemming
        tweet = [stemmer.stem(word) for word in nltk.word_tokenize(tweet) if word not in stops]
        # lowercase & join again
        return " ".join(tweet).lower()
    
    data = pd.read_csv(dataset_path)
    
    # map correct labels
    data["label"] = pd.Categorical(data["violence"], categories=['no', 'violence', 'malpractice']).codes
    data["label"] = np.where(data["label"] == 2, 0, data["label"])
    
    # lowercase, remove stopwords, remove users
    data["text_original"] = data["text"]
    data["text"] = data["text"].apply(lambda text: preprocess_tweet(text, stops, stemmer))
    
    return data
    
def load_data(data, seed):
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.2, stratify=data["label"], random_state=seed)

    # Do tf-idf vectorization on all sets (fit on train, transform on all)
    tfidf_vectorizer = TfidfVectorizer()
    X_train_vec = tfidf_vectorizer.fit_transform(X_train)
    X_test_vec = tfidf_vectorizer.transform(X_test)
    
    return X_train_vec, X_test_vec, y_train, y_test

def run_svc(X_train_tfidf, X_test_tfidf, y_train, y_test):
    # Define initial SVC model
    svc = SVC(random_state=20211010)

    # Define pipeline for tuning grid
    pipeline = Pipeline(steps=[("svc", svc)])

    # Define parameter combinations for C, gamma and kernel (ignore gamma parameter for linear kernel as it is not availbale for the linear kernel)
    param_grid = [
        {
            "svc__C": np.exp(list(range(0, 11))),
            "svc__gamma": [0.0001, 0.001, 0.01, 0.1, 1, "scale", "auto"],
            "svc__kernel": ["rbf", "poly", "sigmoid"],
            "svc__class_weight": [None, "balanced"]
        },
        {
            "svc__C": np.exp(list(range(0, 11))),
            "svc__kernel": ["linear"],
            "svc__class_weight": [None, "balanced"]
        }
    ]

    # specify the cross validation
    stratified_5_fold_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20211010)

    # Perform grid search on isolated validation set
    grid_search = GridSearchCV(pipeline, param_grid, cv=stratified_5_fold_cv, verbose=2, n_jobs=-1, scoring="f1")
    grid_search.fit(X_train_tfidf, y_train)

    # get the best parameter setting
    print(
        "Best Tuning Score is {} with params {}".format(grid_search.best_score_,
                                                        grid_search.best_params_))

    if grid_search.best_params_["svc__kernel"] == "linear":
        grid_search.best_params_["svc__gamma"] = None

    best_svc = SVC(C=grid_search.best_params_["svc__C"],
                   gamma=grid_search.best_params_["svc__gamma"],
                   kernel=grid_search.best_params_["svc__kernel"],
                   class_weight=grid_search.best_params_["svc__class_weight"],
                   random_state=20211010)
    best_svc.fit(X_train_tfidf, y_train)

    predictions = best_svc.predict(X_test_tfidf)
    test_accuracy = metrics.accuracy_score(y_test, predictions)
    test_precision = metrics.precision_score(y_test, predictions, average='binary')
    test_recall = metrics.recall_score(y_test, predictions, average='binary')
    test_f1 = metrics.f1_score(y_test, predictions, average='binary')

    return [test_accuracy, test_precision, test_recall, test_f1], [grid_search.best_params_["svc__kernel"], grid_search.best_params_["svc__C"], grid_search.best_params_["svc__class_weight"], grid_search.best_params_["svc__gamma"], grid_search.best_score_]


def print_stats(labels, predictions, model="Not specified", country=None):

    '''
    This function evaluates a model and saves the results in an Array
    '''

    results = [model,
            accuracy_score(labels, predictions),
            precision_score(labels, predictions),
            recall_score(labels, predictions),
            f1_score(labels, predictions)]

    return results

def embedding_transform(data, w2v, words, seed):
    #Split data
    X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.2, stratify=data["label"], random_state=seed)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    #Tokenize X_train
    X_train_tokens = []
    for i in X_train:
        X_train_tokens.append(i.split())

    #transform text data into vectors
    def vectorize(sentence):
        vector = [w2v[word] for word in sentence.split() if word in words]
        return vector

    #Transform X_test
    X_train_vec = list(X_train.apply(lambda sentence: vectorize(sentence)))
    X_test_vec = list(X_test.apply(lambda sentence: vectorize(sentence)))

    #Get maximum sentence length
    lens=[]

    for sentence in X_train_vec:
        lens.append(len(sentence))
    for sentence in X_test_vec:
        lens.append(len(sentence))
    max_len = max(lens)

    embedding_dim = len(X_train_vec[0][0])

    for sentence in X_train_vec:
        if len(sentence)<=max_len:
            padding = max_len - len(sentence)
            for i in range(padding):
                sentence.append([0]*embedding_dim)

    for sentence in X_test_vec:
        if len(sentence)<=max_len:
            padding = max_len - len(sentence)
            for i in range(padding):
                sentence.append(np.array([0]*embedding_dim))

    X_train_vec = np.array(X_train_vec)
    X_test_vec = np.array(X_test_vec)

    # Do tf-idf vectorization on all sets (fit on train, transform on all)
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    return X_train_vec, X_train_tfidf, X_test_vec, X_test_tfidf, y_train, y_test

def run_3filter_CNN(X_train, X_test, y_train, y_test, CNN_params, country, param_setting="default"):

    '''

 RUNS A CONVOLUTIONAL NEURAL NETWORK WITH PARALLEL RUNNING FILTERS FOR TEXT CLASSIFICATION

    '''

    ratio_1 = 1.0 - len(y_train[y_train == 1]) / float(len(y_train))  ## ratio of violence instances
    ratio_0 = 1.0 - ratio_1

    class_weight = {0: ratio_0, 1: ratio_1}

    #Special kernel size for venezuela in parameter configuration "rep_settings.py"
    if country == "Venezuela" and CNN_params["name"]=="rep_settings.py":
        kernels = [1,2,3]
    else:
        kernels = CNN_params["kernel"]

    print(datetime.now().strftime("%H:%M:%S"), ": Start Training CNN")

    n_words = X_train.shape[1]
    n_embeddings = X_train.shape[2]

    model1 = Sequential()
    model1.add(Input(shape=(n_words,n_embeddings,1)))
    model1.add(
        Conv2D(filters=CNN_params["filters"], kernel_size=(kernels[0], n_embeddings), activation='relu'))
    model1.add(GlobalMaxPooling2D())

    model2 = Sequential()
    model2.add(Input(shape=(n_words,n_embeddings,1)))
    model2.add(
        Conv2D(filters=CNN_params["filters"], kernel_size=(kernels[1], n_embeddings), activation='relu'))
    model2.add(GlobalMaxPooling2D())

    model3 = Sequential()
    model3.add(Input(shape=(n_words,n_embeddings,1)))
    model3.add(
        Conv2D(filters=CNN_params["filters"], kernel_size=(kernels[2], n_embeddings), activation='relu'))
    model3.add(GlobalMaxPooling2D())

    # model_concat = concatenate([model1.output, model2.output, model3.output], axis=3)
    model_concat = concatenate([model1.output, model2.output, model3.output])
    # # model_concat = MaxPooling1D()(model_concat)
    model_concat = Dropout(CNN_params["dropout"])(model_concat)
    # # model_concat = Flatten()(model_concat)
    model_concat = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.L2(CNN_params["l2_reg_lambda"]))(model_concat)
    model = Model(inputs=[model1.input, model2.input, model3.input], outputs=model_concat)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.AUC()])

    #fit model
    model.fit([X_train, X_train, X_train], y_train, epochs=CNN_params["epochs"], batch_size=CNN_params["batch_size"], class_weight=class_weight)

    #classify sequences
    y_pred = model.predict([X_test, X_test, X_test])
    y_pred =(y_pred>0.5)

    #print results
    results = print_stats(y_test, y_pred, model = "{c}_CNN_{p}".format(p=param_setting, c=country))
    return results

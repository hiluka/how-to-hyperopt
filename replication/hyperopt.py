# IMPORTS
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import keras.backend as K
import keras_tuner

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling1D, GlobalMaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from keras_tuner import HyperModel, Objective
from keras_tuner.tuners import RandomSearch, Hyperband
from keras.models import Sequential, Model
from keras.layers import Dense, concatenate
from keras.wrappers.scikit_learn import KerasClassifier

from gensim import models
import gensim.downloader as api
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer, SpanishStemmer

from functions import *

os.environ["CUDA_VISIBLE_DEVICES"] = ""
seed=2022

#load pre-trained model for correct language
w2v_es = models.KeyedVectors.load_word2vec_format('sbw_vectors.bin', binary=True)
w2v_en = api.load("word2vec-google-news-300")

# initialize dictionary for countries/datasets
countries = {"Venezuela": "raw/vz-tweets.csv"}
                               
#loop over all countries
for country, path in countries.items():

    print("\nCurrent Country: " + country)

    # initialize stopwords and stemmer in correct language
    stops = set(stopwords.words("spanish")) if country == "Venezuela" else set(stopwords.words("english"))
    stemmer = SpanishStemmer() if country == "Venezuela" else EnglishStemmer()
    results_current = []

    #load pre-trained model for correct language, to-do: aus der schleife holen
    if country == "Venezuela":
        w2v = w2v_es
    else:
        w2v = w2v_en

    #preprocess the data
    data = preprocess_data(path, stops, stemmer)

    words=list(w2v.index_to_key)
    vocab_len = len(w2v)

    #iterate through different random train test splits to capture model variation
    X_train_vec, X_train_tfidf, \
    X_test_vec, X_test_tfidf, \
    y_train, y_test = embedding_transform(data, w2v, words, seed)                                 
                                 
    # Tune model                               
    tune_model(X_train_vec, y_train, model= country, runs=500, epochs=200)
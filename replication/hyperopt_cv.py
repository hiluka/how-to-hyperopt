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

#load pre-trained model for correct language
w2v_es = models.KeyedVectors.load_word2vec_format('sbw_vectors.bin', binary=True)
w2v_en = api.load("word2vec-google-news-300")

# initialize dictionary for countries/datasets
countries = {"Ghana": "raw/gh-tweets.csv", "Philippines": "raw/ph-tweets.csv", "Venezuela": "raw/vz-tweets.csv"}

#run several times with different param settings and seeds
seeds = [20210101, 20210102, 20210103]

#initialize result objects
results_df = pd.DataFrame()
results = []

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
    
    i=1

    for seed in seeds:

        print("Run {i}/3".format(i=i))

        #iterate through different random train test splits to capture model variation
        X_train_vec, X_train_tfidf, \
        X_test_vec, X_test_tfidf, \
        y_train_vec, y_test = embedding_transform(data, w2v,words, seed)
        
        tuner = tune_model_cv(X_train_vec, y_train_vec, model= country+str(seed), runs=50, epochs=200)
        
        #build model with best params
        best_hp = tuner.get_best_hyperparameters()[0]
        model = tuner.hypermodel.build(best_hp)
        
        #set class weight
        ratio_1 = 1.0 - len(y_train_vec[y_train_vec == 1]) / float(len(y_train_vec))  ## ratio of violence instances
        ratio_0 = 1.0 - ratio_1
        class_weight = {0: ratio_0, 1: ratio_1}

        #fit model
        model.fit([X_train_vec, X_train_vec, X_train_vec], y_train_vec, epochs=200, batch_size=64, class_weight=class_weight)

        #classify sequences
        y_pred = model.predict([X_test_vec, X_test_vec, X_test_vec])
        y_pred =(y_pred>0.5)

        results.append(print_stats(y_test, y_pred, model = "{c}_CNN_{p}".format(p="tuned params", c=country)))
        pd.DataFrame(results, columns=["model", "accuracy", "precision", "recall", "f1"]).to_csv("temp_results_{c}_run_{p}.csv".format(p=i, c=country))
            
        print(results)
        
        i+=1
        
#combine all results and calculate summary statistics
results_df = pd.DataFrame(results, columns=["model", "accuracy", "precision", "recall", "f1"])
results_df.to_pickle("results.pkl")
cnn_results = results_df.groupby(results_df["model"]).agg([np.mean, np.std])
cnn_results.to_csv("final_results.csv")

print(cnn_results)


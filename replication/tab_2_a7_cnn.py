#--------------------------------------------------------------------------------
# The Role of Hyperparameters in Machine Learning Models and How to Tune Them (PSRM, 2023)
# Christian Arnold, Luka Biedebach, Andreas Küpfer, and Marcel Neunhoeffer
#--------------------------------------------------------------------------------

#IMPORTS for Baselines
from functions import load_data, preprocess_data, run_svc, run_dummy, run_randomforest, run_naivebayes
import numpy as np
np.set_printoptions(precision=15)
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer, SpanishStemmer

import nltk
nltk.download('stopwords')
nltk.download('punkt')

# IMPORTS for CNN
import os
import tensorflow as tf
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

from functions import *

import argparse

parser = argparse.ArgumentParser(description='"Run CNN code.')
parser.add_argument('--rerun', action='store_true', help='Rerun tuning completely')
args = parser.parse_args()
# Set to True if models should be run again
rerun = args.rerun

if rerun:
    
    print("Rerun flag true: tuning in progress...")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    #load pre-trained model for correct language
    w2v_es = models.KeyedVectors.load_word2vec_format('sbw_vectors.bin', binary=True)
    w2v_en = api.load("word2vec-google-news-300")


    # initialize dictionary for countries/datasets
    countries = {"Ghana": "data/gh-tweets_full.csv", "Philippines": "data/ph-tweets_full.csv", "Venezuela": "data/vz-tweets_full.csv"}

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

        #load pre-trained model for correct language
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
            
            #tune model
            tuner = tune_model_cv(X_train_vec, y_train_vec, model= country+str(seed), runs=50, epochs=200)

            #build model with best params
            best_hp = tuner.get_best_hyperparameters()[0]
            model = tuner.hypermodel.build(best_hp)

            #set class weight
            ratio_1 = 1.0 - len(y_train_vec[y_train_vec == 1]) / float(len(y_train_vec))  ## ratio of violence instances
            ratio_0 = 1.0 - ratio_1
            class_weight = {0: ratio_0, 1: ratio_1}

            #fit model
            model.fit([X_train_vec, X_train_vec, X_train_vec], y_train_vec, epochs=200, batch_size=64, class_weight=class_weight, verbose=0)

            #classify sequences
            y_pred = model.predict([X_test_vec, X_test_vec, X_test_vec])
            y_pred =(y_pred>0.5)

            #save tuned results
            r = print_stats(y_test, y_pred, model = "CNN_tuned")
            r.extend([country, seed, best_hp.values["filters"], [best_hp.values["kernel"],best_hp.values["kernel"]+1, best_hp.values["kernel"]+2] , best_hp.values["dropout"], best_hp.values["l2_reg_lambda"], best_hp.values["learning_rate"]])
            results.append(r)

            #default parameters
            default_param = {"name": "default",
                            "batch_size":64,
                            "epochs":200,
                            "filters":200,
                            "kernel":[1,2,3],
                            "dropout":0.5,
                            "l2_reg_lambda": 0.01,
                            "learning_rate": 0.001}

            #train one CNN model on the "default" values
            y_pred = run_CNN(X_train_vec, X_test_vec, y_train_vec, y_test, default_param, param_setting="default", country=country)

            # save default result
            d =  print_stats(y_test, y_pred, model = "CNN_default")
            d.extend([country, seed, default_param["filters"], default_param["kernel"], default_param["dropout"], default_param["l2_reg_lambda"], default_param["learning_rate"]])
            results.append(d)

            #save temporary results
            results_df_temp = pd.DataFrame(results, columns=['model', 'accuracy', 'precision', 'recall', 'f1', 'country', 'seed','filters', 'kernel', 'dropout', 'l2_reg_lambda', 'learning_rate']).round(3)
            results_df_temp.to_csv("results/cnn_temp_{c}_{s}.csv".format(c=country, s=seed))

            i+=1

    #combine all results and calculate summary statistics
    a7 = pd.DataFrame(results, columns=['model', 'accuracy', 'precision', 'recall', 'f1', 'country', 'seed','filters', 'kernel', 'dropout', 'l2_reg_lambda', 'learning_rate']).round(3)
    a7.to_csv("results/cnn_results_a7.csv", index=False)

if not rerun:
    
    print("Rerun flag false: results loaded from files...")
    a7 = pd.read_csv("results/cnn_results_a7.csv")
    

a7["tuning"] = a7["model"].apply(lambda x: x.split("_")[1])
tab2 = a7[["country", "tuning", "f1"]][a7["seed"]==20210101]
tab2.to_csv("results/cnn_results_tab2.csv", index=False)

## Print and store detailed result scores
print("--- Begin Table 2 ---")
print(tab2)
print("--- End Table 2 ---")

print("--- Begin Table A7 ---")
print(a7)
print("--- End Table A7 ---")

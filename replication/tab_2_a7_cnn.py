#--------------------------------------------------------------------------------
# The Role of Hyperparameters in Machine Learning Models and How to Tune Them (PSRM, 2023)
# Christian Arnold, Luka Biedebach, Andreas Küpfer, and Marcel Neunhoeffer
#--------------------------------------------------------------------------------

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

from utils.functions import *

import argparse

parser = argparse.ArgumentParser(description='"Run CNN code.')
parser.add_argument('--rerun', type=bool, default=False, help='Rerun flag (default: False)')
args = parser.parse_args()
# Set to True if models should be run again
rerun = args.rerun

if rerun:
    print("Rerun flag true: tuning in progress...")
if not rerun:
    print("Rerun flag false: results loaded from files...")

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
        
        tuner = tune_model_cv(X_train_vec, y_train_vec, model= country+str(seed), runs=2, epochs=200)
        
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

        results.append(best_hp+[print_stats(y_test, y_pred, model = "{c}_CNN_{p}".format(p="tuned params", c=country))])
        pd.DataFrame(results, columns=["parameters", "model", "accuracy", "precision", "recall", "f1"]).to_csv("temp_results_{c}_run_{p}.csv".format(p=i, c=country))
            
        print(results)
        
        i+=1
        
#combine all results and calculate summary statistics
results_df = pd.DataFrame(results, columns=["parameters", "model", "accuracy", "precision", "recall", "f1"])
cnn_results.to_csv("results.csv")
cnn_results = results_df.groupby(results_df["model"]).agg([np.mean, np.std])
cnn_results.to_csv("final_results.csv")

if not rerun:
    results_cnn = pd.read_csv("results/cnn_results.csv")

## Print and store detailed result scores
print("--- Begin Table 2 ---")
# Here CNN results
print("--- End Table 2 ---")
results_cnn.round(3).to_csv("results/cnn_results.csv", index=False)

if rerun:
    results_cnn_full.round(3).to_csv("results/cnn_results_full.csv", index=False)

if not rerun:
    results_tuning_svc = pd.read_csv("results/cnn_results_hyperparameter.csv")

print("--- Begin Table A7 ---")
print(results_tuning_cnn.round(3))
print("--- End Table A7 ---")

results_tuning_cnn.round(3).to_csv("results/cnn_results_hyperparameter.csv", index=False)

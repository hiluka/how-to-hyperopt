#IMPORTS
from gensim import models
import gensim.downloader as api
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer, SpanishStemmer

from functions import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

#Define Parameter Settings
manuscript_params = {"name": "Manuscript",
                     "batch_size":64,  
                    "epochs":200,
                    "filters":200,
                    "kernel":[3,4,5],
                    "dropout":0.5,
                    "l2_reg_lambda": 0.01,
                    "window_size": 5,
                    "learning_rate": 0.001}

cnn_train_params = {"name": "cnn_train.py",
                    "batch_size":64,
                    "epochs":200,
                    "filters":200,
                    "kernel":[1,2,3],
                    "dropout":0.5,
                    "l2_reg_lambda": 0.01,
                    "window_size": 10,
                    "learning_rate": 0.001}

rep_settings_params = {"name": "rep_settings.py",
                       "batch_size":64,
                        "epochs":200,
                        "filters":200,
                        "kernel":[2,3,4],
                        "dropout":0.8,
                        "l2_reg_lambda": 0.001,
                        "window_size": 5,
                        "learning_rate": 0.001}


default_params = {"name": "default",
                       "batch_size":32,
                        "epochs":200,
                        "filters":200,
                        "kernel":[3,3,3],
                        "dropout":0,
                        "l2_reg_lambda": 0.01,
                        "window_size": 5,
                        "learning_rate": 0.001}

#load pre-trained model for correct language
w2v_es = models.KeyedVectors.load_word2vec_format('sbw_vectors.bin', binary=True)
w2v_en = api.load("word2vec-google-news-300")

# initialize dictionary for countries/datasets
countries = {"Venezuela": "raw/vz-tweets.csv", "Ghana": "raw/gh-tweets.csv", "Philippines": "raw/ph-tweets.csv"}

#run several times with different param settings and seeds
seeds = [20210101, 20210102, 20210103, 20210104, 20210105]
params = [manuscript_params, cnn_train_params, rep_settings_params]

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

        print("Run {i}/5".format(i=i))

        #iterate through different random train test splits to capture model variation
        X_train_vec, X_train_tfidf, \
        X_test_vec, X_test_tfidf, \
        y_train, y_test = embedding_transform(data,
                                               w2v,
                                               words,
                                               seed)

        #run CNN in 3 different configurations
        for param in params:
            results.append(run_3filter_CNN(X_train_vec, X_test_vec, y_train, y_test, param, param_setting=param["name"], country=country))
        
        print(results)
        i+=1
        
#combine all results and calculate summary statistics
results_df = pd.DataFrame(results, columns=["model", "accuracy", "precision", "recall", "f1"])
results_df.to_pickle("results.pkl")
cnn_results = results_df.groupby(results_df["model"]).agg([np.mean, np.std])

print(cnn_results)

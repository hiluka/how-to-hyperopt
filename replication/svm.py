from functions import load_data, preprocess_data, run_svc
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer, SpanishStemmer

# set 5 different seeds for reproducibility
seeds = [20210101, 20210102, 20210103, 20210104, 20210105]

# initialize dictionary for countries/datasets
countries = {"Venezuela": "raw/vz-tweets.csv", "Ghana": "raw/gh-tweets.csv", "Philippines": "raw/ph-tweets.csv"}

# define dataframe to store results
results = pd.DataFrame(
    columns=["Country", "Accuracy", "Accuracy Std. Dev.", "Precision", "Precision Std. Dev.", "Recall",
             "Recall Std. Dev.", "F1", "F1 Std. Dev."])
results_tuning = pd.DataFrame(columns=["Country", "kernel", "C", "class_weight", "gamma", "Tuning F1"])

# loop over all countries
for country, path in countries.items():
    print("\nCurrent Country: " + country)
    # initialize stopwords and stemmer in correct language
    stops = set(stopwords.words("spanish")) if country == "Venezuela" else set(stopwords.words("english"))
    stemmer = SpanishStemmer() if country == "Venezuela" else EnglishStemmer()
    results_scores_current = []
    results_tuning_current = []

    # preprocess the data
    data = preprocess_data(path, stops, stemmer)

    # loop over seeds, load data and tune/train SVC
    for seed in seeds:
        X_train_tfidf, X_test_tfidf, y_train, y_test = load_data(data, seed)
        result_scores, results_tuning_current = run_svc(X_train_tfidf, X_test_tfidf, y_train, y_test)
        results_scores_current.append(result_scores)
        results_tuning = results_tuning.append({"Country": country,
                                                "kernel": results_tuning_current[0],
                                                "C": results_tuning_current[1],
                                                "class_weight": results_tuning_current[2],
                                                "gamma": results_tuning_current[3],
                                                "Tuning F1": results_tuning_current[4]}, ignore_index=True)

    # calculate means/standard deviations from results
    results_current_mean = np.array(results_scores_current).mean(axis=0)
    results_std_dev = np.array(results_scores_current).std(axis=0)
    results = results.append({"Country": country,
                              "Accuracy": results_current_mean[0],
                              "Accuracy Std. Dev.": results_std_dev[0],
                              "Precision": results_current_mean[1],
                              "Precision Std. Dev.": results_std_dev[1],
                              "Recall": results_current_mean[2],
                              "Recall Std. Dev.": results_std_dev[2],
                              "F1": results_current_mean[3],
                              "F1 Std. Dev.": results_std_dev[3]}, ignore_index=True)

# Store detailed result scores
print(results)
results.round(3).to_csv("svm_results_tuned.csv", index=False)

# Store best hyperparameter combinations (5 tuning runs) for each country
print(results_tuning)
results_tuning.round(3).to_csv("svm_results_hyperparameter.csv", index=False)
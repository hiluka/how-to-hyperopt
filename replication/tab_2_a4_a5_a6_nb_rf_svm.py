from utils.functions import *
import numpy as np
np.set_printoptions(precision=15)
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer, SpanishStemmer

import nltk
nltk.download('stopwords')

import argparse

parser = argparse.ArgumentParser(description='"Run Naive Bayes, Random Forest, and Support Vector Machine code.')
parser.add_argument('--rerun', type=bool, default=False, help='Rerun flag (default: False)')
args = parser.parse_args()
rerun = args.rerun

if rerun:
    print("Rerun flag true: tuning in progress...")
if not rerun:
    print("Rerun flag false: results loaded from files...")

# set 5 different seeds for reproducibility
seeds = [20210101, 20210102, 20210103, 20210104, 20210105]

# initialize dictionary for countries/datasets
countries = {"Venezuela": "raw/vz-tweets_full.csv", "Ghana": "raw/gh-tweets_full.csv", "Philippines": "raw/ph-tweets_full.csv"}

# define dataframe to store results
results_svc = pd.DataFrame(
    columns=["Baseline", "Country", "Accuracy", "Precision", "Recall", "F1"])
results_svc_full = pd.DataFrame(
    columns=["Baseline", "Country", "Accuracy", "Precision", "Recall", "F1"])

results_randomforest = pd.DataFrame(
    columns=["Baseline", "Country", "Accuracy", "Precision", "Recall", "F1"])
results_randomforest_full = pd.DataFrame(
    columns=["Baseline", "Country", "Accuracy", "Precision", "Recall", "F1"])

results_naivebayes = pd.DataFrame(
    columns=["Baseline", "Country", "Accuracy", "Precision", "Recall", "F1"])
results_naivebayes_full = pd.DataFrame(
    columns=["Baseline", "Country", "Accuracy", "Precision", "Recall", "F1"])

results_tuning_svc = pd.DataFrame(columns=["Country", "kernel", "C", "class_weight", "gamma", "Tuning F1", "OOS F1"])
results_tuning_randomforest = pd.DataFrame(columns=["Country", "max_depth", "n_estimators", "class_weight", "max_features", "Tuning F1", "OOS F1"])
results_tuning_naivebayes = pd.DataFrame(columns=["Country", "alpha", "Tuning F1", "OOS F1"])

# loop over all countries

for country, path in countries.items():
    print("\nCurrent Country: " + country)
    # initialize stopwords and stemmer in correct language
    stops = set(stopwords.words("spanish")) if country == "Venezuela" else set(stopwords.words("english"))
    stemmer = SpanishStemmer() if country == "Venezuela" else EnglishStemmer()
    
    results_scores_current_svc = []
    results_scores_current_randomforest = []
    results_scores_current_naivebayes = []
    results_scores_current_untuned_svc = []
    results_scores_current_untuned_randomforest = []
    results_scores_current_untuned_naivebayes = []
    
    results_tuning_current_svc = []
    results_tuning_current_randomforest = []
    results_tuning_current_naivebayes = []
    
    # preprocess the data
    if rerun:
        data = preprocess_data(path, stops, stemmer)

    # loop over seeds, load data and tune/train baseline models
    for seed in seeds:
        if rerun:
            X_train_tfidf, X_test_tfidf, y_train, y_test = load_data(data, seed)

        # SVC Tuned
        print("SVC...")
        if rerun:
            result_scores, results_tuning_current = run_svc(X_train_tfidf, X_test_tfidf, y_train, y_test)
            results_scores_current_svc.append(result_scores)
            results_tuning_svc = results_tuning_svc.append({"Country": country,
                                                    "kernel": results_tuning_current[0],
                                                    "C": results_tuning_current[1],
                                                    "class_weight": results_tuning_current[2],
                                                    "gamma": results_tuning_current[3],
                                                    "Tuning F1": results_tuning_current[4],
                                                    "OOS F1": result_scores[3]}, ignore_index=True)
            print(result_scores)

        # SVC Untuned
        print("SVC Default...")
        if rerun:
            result_scores = run_svc(X_train_tfidf, X_test_tfidf, y_train, y_test, tune = False)
            results_scores_current_untuned_svc.append(result_scores)
            print(result_scores)
        
        # Random Forest Tuned
        print("Random Forest...")
        if rerun:
            result_scores, results_tuning_current = run_randomforest(X_train_tfidf, X_test_tfidf, y_train, y_test)
            results_scores_current_randomforest.append(result_scores)
            results_tuning_randomforest = results_tuning_randomforest.append({"Country": country,
                                                    "max_depth": results_tuning_current[0],
                                                    "n_estimators": results_tuning_current[1],
                                                    "class_weight": results_tuning_current[2],
                                                    "max_features": results_tuning_current[3],
                                                    "Tuning F1": results_tuning_current[4],
                                                    "OOS F1": result_scores[3]}, ignore_index=True)
            print(result_scores)
        
        # Random Forest Untuned  
        print("Random Forest Default...")
        if rerun:
            result_scores = run_randomforest(X_train_tfidf, X_test_tfidf, y_train, y_test, tune = False)
            results_scores_current_untuned_randomforest.append(result_scores)
            print(result_scores)
        
        # Naives Bayes Tuned
        print("Naives Bayes...")
        if rerun:
            result_scores, results_tuning_current = run_naivebayes(X_train_tfidf, X_test_tfidf, y_train, y_test)
            results_scores_current_naivebayes.append(result_scores)
            results_tuning_naivebayes = results_tuning_naivebayes.append({"Country": country,
                                                                          "alpha": results_tuning_current[0],
                                                                          "Tuning F1": results_tuning_current[1],
                                                                          "OOS F1": result_scores[3]}, ignore_index=True)
            print(result_scores)
        
        # Naives Bayes Untuned  
        print("Naives Bayes Default...")
        if rerun:
            result_scores = run_naivebayes(X_train_tfidf, X_test_tfidf, y_train, y_test, tune = False)
            results_scores_current_untuned_naivebayes.append(result_scores)
            print(result_scores)

    if rerun:
        # SVC Tuned
        results_svc = results_svc.append({"Baseline": "SVM Tuned", "Country": country,
                                  "Accuracy": np.array(results_scores_current_svc)[:, -4][0],
                                  "Precision": np.array(results_scores_current_svc)[:, -3][0],
                                  "Recall": np.array(results_scores_current_svc)[:, -2][0],
                                  "F1": np.array(results_scores_current_svc)[:, -1][0]}, ignore_index=True)
        results_svc_full = results_svc_full.append({"Baseline": "SVM", "Country": country,
                              "Accuracy": results_scores_current_svc,
                              "Precision": results_scores_current_svc,
                              "Recall": results_scores_current_svc,
                              "F1": results_scores_current_svc}, ignore_index=True)

        # SVC Untuned                              
        results_svc = results_svc.append({"Baseline": "SVM Default", "Country": country,
                                  "Accuracy": np.array(results_scores_current_untuned_svc)[:, -4][0],
                                  "Precision": np.array(results_scores_current_untuned_svc)[:, -3][0],
                                  "Recall": np.array(results_scores_current_untuned_svc)[:, -2][0],
                                  "F1": np.array(results_scores_current_untuned_svc)[:, -1][0]}, ignore_index=True)
        results_svc_full = results_svc_full.append({"Baseline": "SVM Untuned", "Country": country,
                              "Accuracy": results_scores_current_untuned_svc,
                              "Precision": results_scores_current_untuned_svc,
                              "Recall": results_scores_current_untuned_svc,
                              "F1": results_scores_current_untuned_svc}, ignore_index=True)

        # Random Forest Tuned
        results_randomforest = results_randomforest.append({"Baseline": "Random Forest Tuned", "Country": country,
                                  "Accuracy": np.array(results_scores_current_randomforest)[:, -4][0],
                                  "Precision": np.array(results_scores_current_randomforest)[:, -3][0],
                                  "Recall": np.array(results_scores_current_randomforest)[:, -2][0],
                                  "F1": np.array(results_scores_current_randomforest)[:, -1][0]}, ignore_index=True)
        results_randomforest_full = results_randomforest_full.append({"Baseline": "Random Forest", "Country": country,
                              "Accuracy": results_scores_current_randomforest,
                              "Precision": results_scores_current_randomforest,
                              "Recall": results_scores_current_randomforest,
                              "F1": results_scores_current_randomforest}, ignore_index=True) 

        # Random Forest Untuned
        results_randomforest = results_randomforest.append({"Baseline": "Random Forest Default", "Country": country,
                                  "Accuracy": np.array(results_scores_current_untuned_randomforest)[:, -4][0],
                                  "Precision": np.array(results_scores_current_untuned_randomforest)[:, -3][0],
                                  "Recall": np.array(results_scores_current_untuned_randomforest)[:, -2][0],
                                  "F1": np.array(results_scores_current_untuned_randomforest)[:, -1][0]}, ignore_index=True)
        results_randomforest_full = results_randomforest_full.append({"Baseline": "Random Forest Untuned", "Country": country,
                              "Accuracy": results_scores_current_untuned_randomforest,
                              "Precision": results_scores_current_untuned_randomforest,
                              "Recall": results_scores_current_untuned_randomforest,
                              "F1": results_scores_current_untuned_randomforest}, ignore_index=True) 

        # Naive Bayes Tuned
        results_naivebayes = results_naivebayes.append({"Baseline": "Naive Bayes Tuned", "Country": country,
                                  "Accuracy": np.array(results_scores_current_naivebayes)[:, -4][0],
                                  "Precision": np.array(results_scores_current_naivebayes)[:, -3][0],
                                  "Recall": np.array(results_scores_current_naivebayes)[:, -2][0],
                                  "F1": np.array(results_scores_current_naivebayes)[:, -1][0]}, ignore_index=True)
        results_naivebayes_full = results_naivebayes_full.append({"Baseline": "Naive Bayes", "Country": country,
                              "Accuracy": results_scores_current_naivebayes,
                              "Precision": results_scores_current_naivebayes,
                              "Recall": results_scores_current_naivebayes,
                              "F1": results_scores_current_naivebayes}, ignore_index=True)    

        # Naive Bayes Untuned
        results_naivebayes = results_naivebayes.append({"Baseline": "Naive Bayes Default", "Country": country,
                                  "Accuracy": np.array(results_scores_current_untuned_naivebayes)[:, -4][0],
                                  "Precision": np.array(results_scores_current_untuned_naivebayes)[:, -3][0],
                                  "Recall": np.array(results_scores_current_untuned_naivebayes)[:, -2][0],
                                  "F1": np.array(results_scores_current_untuned_naivebayes)[:, -1][0]}, ignore_index=True)
        results_naivebayes_full = results_naivebayes_full.append({"Baseline": "Naive Bayes", "Country": country,
                              "Accuracy": results_scores_current_untuned_naivebayes,
                              "Precision": results_scores_current_untuned_naivebayes,
                              "Recall": results_scores_current_untuned_naivebayes,
                              "F1": results_scores_current_untuned_naivebayes}, ignore_index=True)

if not rerun:
    results_svc = pd.read_csv("results/svm_results.csv")
    results_randomforest = pd.read_csv("results/randomforest_results.csv")
    results_naivebayes = pd.read_csv("results/naivebayes_results.csv")

## Print and store detailed result scores
print("--- Begin Table 2 ---")
print(results_naivebayes[["Baseline", "Country", "F1"]].round(3))
print(results_randomforest[["Baseline", "Country", "F1"]].round(3))
print(results_svc[["Baseline", "Country", "F1"]].round(3))
print("--- End Table 2 ---")
results_svc.round(3).to_csv("results/svm_results.csv", index=False)
results_randomforest.round(3).to_csv("results/randomforest_results.csv", index=False)
results_naivebayes.round(3).to_csv("results/naivebayes_results.csv", index=False)

if rerun:
    results_svc_full.round(3).to_csv("results/svm_results_full.csv", index=False)
    results_randomforest_full.round(3).to_csv("results/randomforest_results_full.csv", index=False)
    results_naivebayes_full.round(3).to_csv("results/naivebayes_results_full.csv", index=False)

if not rerun:
    results_tuning_svc = pd.read_csv("results/svm_results_hyperparameter.csv")
    results_tuning_randomforest = pd.read_csv("results/randomforest_results_hyperparameter.csv")
    results_tuning_naivebayes = pd.read_csv("results/naivebayes_results_hyperparameter.csv")
# Print and store best hyperparameter combinations (5 tuning runs) for each country
print("--- Begin Table A4 ---")
print(results_tuning_naivebayes.round(3))
print("--- End Table A4 ---")
print("--- Begin Table A5 ---")
print(results_tuning_svc.round(3))
print("--- End Table A5 ---")
print("--- Begin Table A6 ---")
print(results_tuning_randomforest.round(3))
print("--- End Table A6 ---")

results_tuning_svc.round(3).to_csv("results/svm_results_hyperparameter.csv", index=False)
results_tuning_randomforest.round(3).to_csv("results/randomforest_results_hyperparameter.csv", index=False)
results_tuning_naivebayes.round(3).to_csv("results/naivebayes_results_hyperparameter.csv", index=False)

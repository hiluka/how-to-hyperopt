from functions import load_data, preprocess_data, run_svc, run_dummy, run_randomforest, run_naivebayes
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer, SpanishStemmer

import nltk
nltk.download('stopwords')

# set 5 different seeds for reproducibility
seeds = [20210101, 20210102, 20210103, 20210104, 20210105]

# initialize dictionary for countries/datasets
countries = {"Venezuela": "raw/vz-tweets 2.csv", "Ghana": "raw/gh-tweets 2.csv", "Philippines": "raw/ph-tweets 2.csv"}

# define dataframe to store results
results_svc = pd.DataFrame(
    columns=["Baseline", "Country", "Accuracy", "Accuracy Std. Dev.", "Precision", "Precision Std. Dev.", "Recall",
             "Recall Std. Dev.", "F1", "F1 Std. Dev."])
results_svc_full = pd.DataFrame(
    columns=["Baseline", "Country", "Accuracy", "Precision", "Recall", "F1"])

results_randomforest = pd.DataFrame(
    columns=["Baseline", "Country", "Accuracy", "Accuracy Std. Dev.", "Precision", "Precision Std. Dev.", "Recall",
             "Recall Std. Dev.", "F1", "F1 Std. Dev."])
results_randomforest_full = pd.DataFrame(
    columns=["Baseline", "Country", "Accuracy", "Precision", "Recall", "F1"])

results_naivebayes = pd.DataFrame(
    columns=["Baseline", "Country", "Accuracy", "Accuracy Std. Dev.", "Precision", "Precision Std. Dev.", "Recall",
             "Recall Std. Dev.", "F1", "F1 Std. Dev."])
results_naivebayes_full = pd.DataFrame(
    columns=["Baseline", "Country", "Accuracy", "Precision", "Recall", "F1"])

results_tuning_svc = pd.DataFrame(columns=["Country", "kernel", "C", "class_weight", "gamma", "Tuning F1"])
results_tuning_randomforest = pd.DataFrame(columns=["Country", "max_depth", "n_estimators", "class_weight", "max_features", "Tuning F1"])
results_tuning_naivebayes = pd.DataFrame(columns=["Country", "alpha"])

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
    data = preprocess_data(path, stops, stemmer)

    # loop over seeds, load data and tune/train baseline models
    for seed in seeds:
        X_train_tfidf, X_test_tfidf, y_train, y_test = load_data(data, seed)

        # SVC Tuned
        print("SVC...")
        result_scores, results_tuning_current = run_svc(X_train_tfidf, X_test_tfidf, y_train, y_test)
        results_scores_current_svc.append(result_scores)
        results_tuning_svc = results_tuning_svc.append({"Country": country,
                                                "kernel": results_tuning_current[0],
                                                "C": results_tuning_current[1],
                                                "class_weight": results_tuning_current[2],
                                                "gamma": results_tuning_current[3],
                                                "Tuning F1": results_tuning_current[4]}, ignore_index=True)
        print(result_scores)

        # SVC Untuned
        print("SVC Untuned...")
        result_scores = run_svc(X_train_tfidf, X_test_tfidf, y_train, y_test, tune = False)
        results_scores_current_untuned_svc.append(result_scores)
        print(result_scores)
        
        # Random Forest Tuned
        print("Random Forest...")
        result_scores, results_tuning_current = run_randomforest(X_train_tfidf, X_test_tfidf, y_train, y_test)
        results_scores_current_randomforest.append(result_scores)
        results_tuning_randomforest = results_tuning_randomforest.append({"Country": country,
                                                "max_depth": results_tuning_current[0],
                                                "n_estimators": results_tuning_current[1],
                                                "class_weight": results_tuning_current[2],
                                                "max_features": results_tuning_current[3],
                                                "Tuning F1": results_tuning_current[4]}, ignore_index=True)
        print(result_scores)
        
        # Random Forest Untuned  
        print("Random Forest Untuned...")
        result_scores = run_randomforest(X_train_tfidf, X_test_tfidf, y_train, y_test, tune = False)
        results_scores_current_untuned_randomforest.append(result_scores)
        print(result_scores)
        
        # Naives Bayes Tuned
        print("Naives Bayes...")
        result_scores, results_tuning_current = run_naivebayes(X_train_tfidf, X_test_tfidf, y_train, y_test)
        results_scores_current_naivebayes.append(result_scores)
        results_tuning_naivebayes = results_tuning_naivebayes.append({"Country": country,
                                                                      "alpha": results_tuning_current[0],
                                                                      "Tuning F1": results_tuning_current[1]}, ignore_index=True)
        print(result_scores)
        
        # Naives Bayes Untuned  
        print("Naives Bayes Untuned...")
        result_scores = run_naivebayes(X_train_tfidf, X_test_tfidf, y_train, y_test, tune = False)
        results_scores_current_untuned_naivebayes.append(result_scores)
        print(result_scores)
        
    # calculate means/standard deviations from results    
    # SVC Tuned
    results_current_mean_svc = np.array(results_scores_current_svc).mean(axis=0)
    results_std_dev_svc = np.array(results_scores_current_svc).std(axis=0)
    results_svc = results_svc.append({"Baseline": "SVM", "Country": country,
                              "Accuracy": results_current_mean_svc[0],
                              "Accuracy Std. Dev.": results_std_dev_svc[0],
                              "Precision": results_current_mean_svc[1],
                              "Precision Std. Dev.": results_std_dev_svc[1],
                              "Recall": results_current_mean_svc[2],
                              "Recall Std. Dev.": results_std_dev_svc[2],
                              "F1": results_current_mean_svc[3],
                              "F1 Std. Dev.": results_std_dev_svc[3]}, ignore_index=True)
    results_svc_full = results_svc_full.append({"Baseline": "SVM", "Country": country,
                          "Accuracy": results_scores_current_svc,
                          "Precision": results_scores_current_svc,
                          "Recall": results_scores_current_svc,
                          "F1": results_scores_current_svc}, ignore_index=True)
        
    # SVC Untuned                              
    results_current_mean_untuned_svc = np.array(results_scores_current_untuned_svc).mean(axis=0)
    results_std_dev_untuned_svc = np.array(results_scores_current_untuned_svc).std(axis=0)
    results_svc = results_svc.append({"Baseline": "SVM Untuned", "Country": country,
                              "Accuracy": results_current_mean_untuned_svc[0],
                              "Accuracy Std. Dev.": results_std_dev_untuned_svc[0],
                              "Precision": results_current_mean_untuned_svc[1],
                              "Precision Std. Dev.": results_std_dev_untuned_svc[1],
                              "Recall": results_current_mean_untuned_svc[2],
                              "Recall Std. Dev.": results_std_dev_untuned_svc[2],
                              "F1": results_current_mean_untuned_svc[3],
                              "F1 Std. Dev.": results_std_dev_untuned_svc[3]}, ignore_index=True)
    results_svc_full = results_svc_full.append({"Baseline": "SVM Untuned", "Country": country,
                          "Accuracy": results_scores_current_untuned_svc,
                          "Precision": results_scores_current_untuned_svc,
                          "Recall": results_scores_current_untuned_svc,
                          "F1": results_scores_current_untuned_svc}, ignore_index=True)
        
    # Random Forest Tuned
    results_current_mean_randomforest = np.array(results_scores_current_randomforest).mean(axis=0)
    results_std_dev_randomforest = np.array(results_scores_current_randomforest).std(axis=0)
    results_randomforest = results_randomforest.append({"Baseline": "Random Forest", "Country": country,
                              "Accuracy": results_current_mean_randomforest[0],
                              "Accuracy Std. Dev.": results_std_dev_randomforest[0],
                              "Precision": results_current_mean_randomforest[1],
                              "Precision Std. Dev.": results_std_dev_randomforest[1],
                              "Recall": results_current_mean_randomforest[2],
                              "Recall Std. Dev.": results_std_dev_randomforest[2],
                              "F1": results_current_mean_randomforest[3],
                              "F1 Std. Dev.": results_std_dev_randomforest[3]}, ignore_index=True)
    results_randomforest_full = results_randomforest_full.append({"Baseline": "Random Forest", "Country": country,
                          "Accuracy": results_scores_current_randomforest,
                          "Precision": results_scores_current_randomforest,
                          "Recall": results_scores_current_randomforest,
                          "F1": results_scores_current_randomforest}, ignore_index=True) 

    # Random Forest Untuned
    results_current_mean_untuned_randomforest = np.array(results_scores_current_untuned_randomforest).mean(axis=0)
    results_std_dev_untuned_randomforest = np.array(results_scores_current_untuned_randomforest).std(axis=0)
    results_randomforest = results_randomforest.append({"Baseline": "Random Forest Untuned", "Country": country,
                              "Accuracy": results_current_mean_untuned_randomforest[0],
                              "Accuracy Std. Dev.": results_std_dev_untuned_randomforest[0],
                              "Precision": results_current_mean_untuned_randomforest[1],
                              "Precision Std. Dev.": results_std_dev_untuned_randomforest[1],
                              "Recall": results_current_mean_untuned_randomforest[2],
                              "Recall Std. Dev.": results_std_dev_untuned_randomforest[2],
                              "F1": results_current_mean_untuned_randomforest[3],
                              "F1 Std. Dev.": results_std_dev_untuned_randomforest[3]}, ignore_index=True)
    results_randomforest_full = results_randomforest_full.append({"Baseline": "Random Forest Untuned", "Country": country,
                          "Accuracy": results_scores_current_untuned_randomforest,
                          "Precision": results_scores_current_untuned_randomforest,
                          "Recall": results_scores_current_untuned_randomforest,
                          "F1": results_scores_current_untuned_randomforest}, ignore_index=True) 
        
    # Naive Bayes Tuned
    results_current_mean_naivebayes = np.array(results_scores_current_naivebayes).mean(axis=0)
    results_std_dev_naivebayes = np.array(results_scores_current_naivebayes).std(axis=0)
    results_naivebayes = results_naivebayes.append({"Baseline": "Naive Bayes", "Country": country,
                              "Accuracy": results_current_mean_naivebayes[0],
                              "Accuracy Std. Dev.": results_std_dev_naivebayes[0],
                              "Precision": results_current_mean_naivebayes[1],
                              "Precision Std. Dev.": results_std_dev_naivebayes[1],
                              "Recall": results_current_mean_naivebayes[2],
                              "Recall Std. Dev.": results_std_dev_naivebayes[2],
                              "F1": results_current_mean_naivebayes[3],
                              "F1 Std. Dev.": results_std_dev_naivebayes[3]}, ignore_index=True)
    results_naivebayes_full = results_naivebayes_full.append({"Baseline": "Naive Bayes", "Country": country,
                          "Accuracy": results_scores_current_naivebayes,
                          "Precision": results_scores_current_naivebayes,
                          "Recall": results_scores_current_naivebayes,
                          "F1": results_scores_current_naivebayes}, ignore_index=True)    
                              
    # Naive Bayes Untuned
    results_current_mean_untuned_naivebayes = np.array(results_scores_current_untuned_naivebayes).mean(axis=0)
    results_std_dev_untuned_naivebayes = np.array(results_scores_current_untuned_naivebayes).std(axis=0)
    results_naivebayes = results_naivebayes.append({"Baseline": "Naive Bayes Untuned", "Country": country,
                              "Accuracy": results_current_mean_untuned_naivebayes[0],
                              "Accuracy Std. Dev.": results_std_dev_untuned_naivebayes[0],
                              "Precision": results_current_mean_untuned_naivebayes[1],
                              "Precision Std. Dev.": results_std_dev_untuned_naivebayes[1],
                              "Recall": results_current_mean_untuned_naivebayes[2],
                              "Recall Std. Dev.": results_std_dev_untuned_naivebayes[2],
                              "F1": results_current_mean_untuned_naivebayes[3],
                              "F1 Std. Dev.": results_std_dev_untuned_naivebayes[3]}, ignore_index=True)
    results_naivebayes_full = results_naivebayes_full.append({"Baseline": "Naive Bayes Untuned", "Country": country,
                          "Accuracy": results_scores_current_untuned_naivebayes,
                          "Precision": results_scores_current_untuned_naivebayes,
                          "Recall": results_scores_current_untuned_naivebayes,
                          "F1": results_scores_current_untuned_naivebayes}, ignore_index=True) 

## Store detailed result scores
print(results_svc)
print(results_randomforest)
print(results_naivebayes)
results_svc.round(3).to_csv("svm_results.csv", index=False)
results_randomforest.round(3).to_csv("randomforest_results.csv", index=False)
results_naivebayes.round(3).to_csv("naivebayes_results.csv", index=False)

results_svc_full.round(3).to_csv("svm_results_full.csv", index=False)
results_randomforest_full.round(3).to_csv("randomforest_results_full.csv", index=False)
results_naivebayes_full.round(3).to_csv("naivebayes_results_full.csv", index=False)

# Store best hyperparameter combinations (5 tuning runs) for each country
print(results_tuning_svc)
print(results_tuning_randomforest)
print(results_tuning_naivebayes)
results_tuning_svc.round(3).to_csv("svm_results_hyperparameter.csv", index=False)
results_tuning_randomforest.round(3).to_csv("randomforest_results_hyperparameter.csv", index=False)
results_tuning_naivebayes.round(3).to_csv("naivebayes_results_hyperparameter.csv", index=False)

# Optional: Print min/max values per seed
#results = pd.concat([results_svc_full, results_randomforest_full, results_naivebayes_full])
#results['Accuracy_min'] = results.Accuracy.map(lambda x: min(x[0]))
#results['Precision_min'] = results.Accuracy.map(lambda x: min(x[1]))
#results['Recall_min'] = results.Accuracy.map(lambda x: min(x[2]))
#results['F1_min'] = results.Accuracy.map(lambda x: min(x[3]))
#results['Accuracy_max'] = results.Accuracy.map(lambda x: max(x[0]))
#results['Precision_max'] = results.Accuracy.map(lambda x: max(x[1]))
#results['Recall_max'] = results.Accuracy.map(lambda x: max(x[2]))
#results['F1_max'] = results.Accuracy.map(lambda x: max(x[3]))

#results.drop(['Accuracy', 'Precision', 'Recall', 'F1'], axis=1).sort_values(by=["Country"]).reset_index(drop = True).to_csv("min_max_results.csv", index = False)

#print(results.sort_values(by=["Country", "F1"]))
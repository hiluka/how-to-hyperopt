# how-to-hyperopt

This repository contains the code necessary to replicate the findings documented in "The Role of Hyperparameters in Machine Learning Models and How to Tune Them"

#### Contents

This dataverse is structured as follows:
- replication
	- data - Folder containing the dataset used for evaluating hyperparameter tuning
		- gh-tweets.csv - tweet IDs and labels of tweets belonging to Ghana
		- ph-tweets.csv - tweet IDs and labels of tweets belonging to Philippines
		- vz-tweets.csv - tweet IDs and labels of tweets belonging to Venezuela
    		- annotations_march22_2023.csv - paper annotations
	- functions.py - Python file, contains Python helper functions
	- environment.yml - YML-file containing the Python packages with versions for the code based on Python 
	- fig_1_tab_1.Rmd - RMarkdown notebook; replicates figure 1 and table 1
	- tab_2_A4_A5_A6.ipynb - Python notebook; replicates information depicted in table 2, A4, A5, and A6
	- results - Folder containing the .csv files with detailed results of our models
		- svm_results.csv - csv-file containing the best results of our SVM model (tuned vs. untuned)
		- randomforest_results.csv - csv-file containing the best results of our Random Forest model (tuned vs. untuned)
		- naivebayes_results.csv - csv-file containing the best results of our Naive Bayes model (tuned vs. untuned)
		- svm_results_full.csv - csv-file containing the all results of all runs of our SVM model
		- randomforest_results_full.csv - csv-file containing the all results of all runs of our Random Forest model
		- naivebayes_results_full.csv - csv-file containing the all results of all runs of our Naive Bayes model
		- svm_results_hyperparameter.csv - csv-file containing the best hyperparameter combination for our SVM model
		- randomforest_results_hyperparameter.csv - csv-file containing the best hyperparameter combination for our Random Forest model
		- naivebayes_results_hyperparameter.csv - csv-file containing the best hyperparameter combination for our Naive Bayes model
	- fig_1_tab_1.pdf - pdf file; produced from running fig_1_tab_1.Rmd
	- tab_2_A4_A5_A6.pdf - pdf file; produced from running  tab_2_A4_A5_A6.ipynb
- Hyperparameter_Optimisation_Guide.ipynb - Guideline notebook with example cases how to tune hyperparameters and assess their fit

#### Environment

To create a clean environment including all dependencies which are needed, execute:

```bash
conda env create -f environment.yml
```

Activate the Environment with:

```bash
conda activate how-to-hyperopt
```

#### Replication

You can find the tweet IDs used by the models in the replication directory. If you want to replicate the study with the exact same data, please get in touch with the authors. However, all tweets which are still available can be downloaded using the official Twitter API and used for a replication.

In order to run cnn.py, please download the pretrained Word2Vec Model for the spanish tweets at https://crscardellino.ar/SBWCE/ and save it in the replication folder.

#### Hyperparameter Optimisation Guide

The file Hyperparameter_Optimisation_Guide.ipynb can be used as a guideline for learning and teaching. It explains how to tune hyperparameters using different data sources and models.

#### Contact

If you have any further questions or encounter issues while replicating the results please let us know via email to Andreas Küpfer (andreas.kuepfer@tu-darmstadt.de). You can also open an issue in our github repository: <https://github.com/hiluka/how-to-hyperopt/issues>

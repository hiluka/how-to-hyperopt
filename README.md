# how-to-hyperopt

This repository contains the code necessary to replicate the tables and figures documented in "The Role of Hyperparameters in Machine Learning Models and How to Tune Them" (Christian Arnold, Luka Biedebach, Andreas Küpfer, Marcel Neunhoeffer). The most current version of the replication archive is hosted on GitHub (https://github.com/hiluka/how-to-hyperopt/), while the Harvard Dataverse is a snapshot of our replication.

#### Contents

This dataverse is structured as follows:
- replication
	- data - Folder containing the dataset used for evaluating hyperparameter tuning
		- gh-tweets.csv - tweet IDs and labels of tweets belonging to Ghana
		- ph-tweets.csv - tweet IDs and labels of tweets belonging to the Philippines
		- vz-tweets.csv - tweet IDs and labels of tweets belonging to Venezuela
		- annotations_march22_2023.csv - paper annotations
	- utils - Folder containing util functions
		- functions.py - Python file, contains Python helper functions
	- environment.yml - YML-file containing the Python packages with versions for the code based on Python 
	- fig_1_tab_1.Rmd - RMarkdown notebook; replicates Figure 1 and Table 1
	- fig_1_tab_1.pdf - pdf file; Log produced from running fig_1_tab_1.Rmd
	- tab_2_A4_A5_A6_A7.ipynb - Python notebook; replicates information depicted in table 2, A4, A5, A6 and A7
	- tab_2_a4_a5_a6_a7.py - Python script which calls tab_2_a4_a5_a6_nb_rf_svm.py and tab_2_a7_cnn.py (set rerun either to True or False within the script)
	- tab_2_a4_a5_a6_nb_rf_svm.txt - Log produced from running tab_2_a4_a5_a6_a7.py
	- tab_2_a7_cnn.txt - Log produced from running tab_2_a4_a5_a6_a7.py
	- tab_2_a4_a5_a6_nb_rf_svm.py - Python Script to replicate Table 2, A4, A5, A6 and A7 in isolation (NB, RF, SVM only)
	- tab_2_a7_cnn.py - Python Script to replicate Table 2 and A7 in isolation (CNN only)
	- results - Folder containing the .csv files with detailed results of our models
		- fig1.png - Figure 1 output (hyperparameter tuning simulation)
		- tab1.csv - Table 1 output (literature study)
		- svm_results.csv - csv-file containing the best results of our SVM model (tuned vs. untuned)
		- randomforest_results.csv - csv-file containing the best results of our Random Forest model (tuned vs. untuned)
		- naivebayes_results.csv - csv-file containing the best results of our Naive Bayes model (tuned vs. untuned)
		- cnn_results.csv - csv-file containing the best results of our CNN model (tuned vs. untuned)
		- svm_results_full.csv - csv-file containing all results of all runs of our SVM model
		- randomforest_results_full.csv - csv-file containing all results of all runs of our Random Forest model
		- naivebayes_results_full.csv - csv-file containing all results of all runs of our Naive Bayes model
		- cnn_results_full.csv - csv-file containing all results of all runs of our CNN model
		- svm_results_hyperparameter.csv - csv-file containing the best hyperparameter combination for our SVM model
		- randomforest_results_hyperparameter.csv - csv-file containing the best hyperparameter combination for our Random Forest model
		- naivebayes_results_hyperparameter.csv - csv-file containing the best hyperparameter combination for our Naive Bayes model
		- cnn_results_hyperparameter.csv - csv-file containing the best hyperparameter combination for our CNN model
- Hyperparameter_Optimisation_Guide.ipynb - Guideline notebook with example cases of how to tune hyperparameters and assess their fit

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

You can find the tweet IDs used by the models in the replication directory. If you want to replicate the study with the exact same data, please get in touch with the authors. However, all tweets which are still available can be downloaded using the official Twitter API and used for replication.
When running the CNN replication (files tab_2_A4_A5_A6.ipynb or tab_2_a4_a5_a6_cnn.py), the pre-trained Word2Vec Model for the Spanish tweets at [https://crscardellino.ar/SBWCE/](https://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors-300-min5.bin.gz) is downloaded automatically.

The most convenient way to replicate all the information depicted in all tables and figures is the following:
- Run RMarkdown file "fig_1_tab_1.Rmd" (Figure 1 and Table 1)
- Create a conda environment for Python dependencies (see above)
- Run Python file "tab_2_a4_a5_a6_a7.py" (Figure 2, A4, A5, A6, and A7)

The combined tuning for all models takes up to 48 hours (tested on a M1 Mac).

#### Hyperparameter Optimisation Guide

The file Hyperparameter_Optimisation_Guide.ipynb can be used as a guideline for learning and teaching. It explains how to tune hyperparameters using different data sources and models.

#### Contact

If you have any further questions or encounter issues while replicating the results, please let us know via email to Andreas Küpfer (andreas.kuepfer@tu-darmstadt.de). You can also open an issue in our GitHub repository: <https://github.com/hiluka/how-to-hyperopt/issues>

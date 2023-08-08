#--------------------------------------------------------------------------------
# The Role of Hyperparameters in Machine Learning Models and How to Tune Them (PSRM, 2023)
# Christian Arnold, Luka Biedebach, Andreas Küpfer, and Marcel Neunhoeffer
#--------------------------------------------------------------------------------

import os
import argparse

# Setup argparse
parser = argparse.ArgumentParser(description='Run model replication')
parser.add_argument('--rerun', action='store_true', help='Rerun tuning completely')
args = parser.parse_args()

# Convert arguments to boolean
rerun = args.rerun

# Table 2, A4, A5, A6, and A7
# Naive Bayes, Random Forest, and Support Vector Machine
os.system(f"python tab_2_a4_a5_a6_nb_rf_svm.py --rerun > tab_2_a4_a5_a6_nb_rf_svm.txt" if rerun else f"python tab_2_a4_a5_a6_nb_rf_svm.py > tab_2_a4_a5_a6_nb_rf_svm.txt")
# CNN
os.system(f"python tab_2_a7_cnn.py --rerun > tab_2_a7_cnn.txt" if rerun else f"python tab_2_a7_cnn.py > tab_2_a7_cnn.txt")

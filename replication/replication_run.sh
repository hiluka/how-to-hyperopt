#!/bin/bash
#--------------------------------------------------------------------------------
# The Role of Hyperparameters in Machine Learning Models and How to Tune Them (PSRM, 2023)
# Christian Arnold, Luka Biedebach, Andreas Küpfer, and Marcel Neunhoeffer
#--------------------------------------------------------------------------------

# Figure 1 
Rscript fig_1.R
# Table 1
Rscript tab_1.R
# Table 2, A4, A5, A6, and A7
python tab_2_a4_a5_a6_nb_rf_svm.py --rerun=False
python tab_2_a7_cnn.py --rerun=False
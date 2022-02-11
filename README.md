# how-to-hyperopt

## Supplementary code for the paper "The Role of Hyperparameters in Machine Learning Models and How to Tune Them"

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

The replication repository is structured as follows:

1) **annotating_ml_papers:** Contains the code and guidelines for the machine learning paper annotation
2) **plots:** Contains the code to create the examplary plots for hyperparameters in regression and svm
3) **replication:** Replicates the code by Muchlinski et al. and shows that optimizing the hyperparameters improves the svm

In order to run cnn.py, please download the pretrained Word2Vec Model for the spanish tweets at https://crscardellino.ar/SBWCE/ and save it in the replication folder.

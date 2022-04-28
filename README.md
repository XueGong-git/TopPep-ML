# AntiCP-Topo

This manual is for the code implementation of paper "Topology based machine learning model for the prediction of anticancer peptides".
# Code Requirements
---
        Platform: Python>=3.6, M
        Python Packages needed: math, numpy>=1.19.5, scipy>=1.4.1, scikit-learn>=0.20.3, pandas>=1.1.3


## Introduction
AntiCP-Topo is developed for predicting, designing and scanning anticancer peptides.


## Abstract
In recent years, interest in the use of therapeutic peptides for treating cancer has grown vastly. A variety of approaches based on machine learning have been explored for anticancer peptide identification while the featurization of these peptides is also critical to attaining any reasonable predictive efficacy using machine learning algorithms. In this paper(repo), we propose three topological-based featurization encodings. Machine learning models were developed using these features on two datasets: main and alternative datasets which were subsequently benchmarked with existing machine learning models. The independent testing results demonstrated that the models developed in this study had marked improvements in accuracy, specificity, and sensitivity to that of the baseline model AntiCP2.0 on both datasets. There is great potential in leveraging topological-based featurization alongside existing feature encoding techniques to accelerate the reliable identification of anticancer peptides for clinical usage.


## AntiCP-Topo Package Files

Brief description of the files given below:
```
LICENSE       	: License information

README.md     	: This file provide information about this package

data            : datasets used taken from https://webs.iiitd.edu.in/raghava/anticp2/download.php

src/construct_dataset.py : Constructing dataframe with features from raw files

src/utils.py    : Documentation of feature encoding methods

src/train.py    : Main Training script to run

```
## Feature generation using SpectPep, magnus & natural representation

For each peptide sequence, the respective feature encoding methods will generate feature vectors for downstream machine learning algorithms.
```python
def L0(args):
    # This function constructs a L0 (Laplacian) symmetric normalised matrix from its boundary matrices (B1).


def L1(args):
    # This function constructs a L1 (Laplacian) symmetric normalised matrix from its boundary matrices (B2).


def agg_magnus_vector(args):
    # Given a input peptide sequence, generates a k-mer magnus vector for each window size and returns an aggregated magnus vector.

def nature_vector(dist):
    # Given a input peptide sequence, generate 3 statistics for each unique amino acid and return all the results in a vector.
```
The above functions can be found in src/utils.py.

## Cite
If you use this code in your research, please cite our paper:

* To be inserted.

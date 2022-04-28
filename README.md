# AntiCP-Topo

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

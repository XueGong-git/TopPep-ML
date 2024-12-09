#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:01:24 2024

@author: gongxue
"""

import itertools
import logging
import Top_ML_mACPpred2

# Set up logging
logging.basicConfig(
    filename='parameter_runs.log',  # Log file name
    level=logging.INFO,            # Log level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
)

# Log the start of the script
logging.info("Starting parameter testing script.")

# Define parameter values
datanames = ["mACPpred2"]  # Add more datanames if needed
classifiers = [ "Etrees"] #, "SVM", "RandomForest"]  # Add more classifiers
scaling_options = [False]
thresholding_models_options = [True]
windows = [5]  # Add more window sizes
iterations = [100]  # Add more iteration counts
feature_sets = [
    ["Mean_magnus", "Natural_Vector",  "L0_ev_avg",  "L1_ev_avg", "N5_natural", "N5_magnus_mean"],
    ["Mean_magnus", "Natural_Vector",  "L0_ev_avg",  "L1_ev_avg", "N10_natural", "N10_magnus_mean"],
    ["Mean_magnus", "Natural_Vector",  "L0_ev_avg",  "L1_ev_avg", "N15_natural", "N15_magnus_mean"],
    ["Mean_magnus", "Natural_Vector",  "L0_ev_avg",  "L1_ev_avg", "C5_natural", "C5_magnus_mean"],
    ["Mean_magnus", "Natural_Vector",  "L0_ev_avg",  "L1_ev_avg", "C10_natural", "C10_magnus_mean"],
    ["Mean_magnus", "Natural_Vector",  "L0_ev_avg",  "L1_ev_avg", "C15_natural", "C15_magnus_mean"],
    ["Mean_magnus", "Natural_Vector",  "L0_ev_avg",  "L1_ev_avg", "N5C5_natural", "N5C5_magnus_mean"],
    ["Mean_magnus", "Natural_Vector",  "L0_ev_avg",  "L1_ev_avg", "N10C10_natural", "N10C10_magnus_mean"],
    ["Mean_magnus", "Natural_Vector",  "L0_ev_avg",  "L1_ev_avg", "N15C15_natural", "N15C15_magnus_mean"],
    
    ["Mean_magnßus", "Natural_Vector", "L0_ev_count", "L1_ev_count",  "N5_natural", "N5_magnus_mean"],
    ["Mean_magnus", "Natural_Vector", "L0_ev_count", "L1_ev_count",  "N10_natural", "N10_magnus_mean"],
    ["Mean_magnus", "Natural_Vector", "L0_ev_count", "L1_ev_count",  "N15_natural", "N15_magnus_mean"],
    ["Mean_magnus", "Natural_Vector", "L0_ev_count", "L1_ev_count",  "C5_natural", "C5_magnus_mean"],
    ["Mean_magnus", "ßNatural_Vector", "L0_ev_count", "L1_ev_count",  "C10_natural", "C10_magnus_mean"],
    ["Mean_magnus", "Natural_Vector", "L0_ev_count", "L1_ev_count",  "C15_natural", "C15_magnus_mean"],
    ["Mean_magnus", "Natural_Vector", "L0_ev_count", "L1_ev_count",  "N5C5_natural", "N5C5_magnus_mean"],
    ["Mean_magnus", "Natural_Vector", "L0_ev_count", "L1_ev_count",  "N10C10_natural", "N10C10_magnus_mean"],
    ["Mean_magnus", "Natural_Vector", "L0_ev_count", "L1_ev_count",  "N15C15_natural", "N15C15_magnus_mean"]

]

# Run combinations of parameters
for dataname, classifier, scaling, thresholding_models, window, iters, final_features in itertools.product(
    datanames, classifiers, scaling_options, thresholding_models_options, windows, iterations, feature_sets
):
    params = f"dataname={dataname}, classifier={classifier}, scaling={scaling}, " \
             f"thresholding_models={thresholding_models}, window={window}, iters={iters}, " \
             f"final_features={final_features}"
    
    logging.info(f"Running with parameters: {params}")
    
    try:
        # Call the main function with these parameters
        Top_ML_mACPpred2.main(
            dataname=dataname,
            classifier=classifier,
            scaling=scaling,
            thresholding_models=thresholding_models,
            window=window,
            iters=iters,
            final_features=final_features,
        )
        logging.info(f"Run completed successfully for parameters: {params}")
    except Exception as e:
        # Log any errors that occur
        logging.error(f"Error occurred with parameters: {params}. Error: {e}")
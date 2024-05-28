import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score

# speed up sklearn
# from sklearnex import patch_sklearn
# patch_sklearn()

# features for alternate dataset (Dataset A)

# final_features = [
#    "Mean_magnus",
#    "Sum_magnus",
#    "Natural_Vector",
#    "L0_ev_avg",
#    "L1_ev_avg",
#    "L0_ev_count",
#    "L1_ev_count",
#    "C15_natural",
#    "C15_magnus_mean"

#    "N5_natural",
#    "N5_magnus_mean"

#    "C5_natural",
#    "C5_magnus_mean"

#    "N5C5_natural",
#    "N5C5_magnus_mean"

#    "N10_natural",
#    "N10_magnus_mean"

#    "C10_natural",
#    "C10_magnus_mean"

#    "N10C10_natural",
#    "N10C10_magnus_mean"

#    "N15_natural",
#    "N15_magnus_mean"

#    "C15_natural",
#    "C15_magnus_mean"


#    "N15C15_natural",
#    "N15C15_magnus_mean"

#]




#print(f"Features used: {final_features}")


# Extra Trees Params
params = {
    "n_estimators": 400,
    "criterion": "gini",
    "n_jobs": -1,
    "max_features": "sqrt",
}

# Random Forest Params
# params = {"n_estimators": 400, "n_jobs": -1, "max_features": "sqrt"}

# GBC Params
#params={'n_estimators': 400,
#        'learning_rate': 0.01,
#        'criterion': 'squared_error',
#        'max_features':'sqrt',
#        'subsample':0.8,
#        'loss': 'log_loss',
#        'max_depth': 8}


iters = 100


def getStandardTime():
    return datetime.today().strftime("%Y-%m-%d-%H_%M")


def train_test_split(df, final_features= None, shuffle=False):
    """
    Prepare the data accordingly based on the given final feature vector size
    and final feature column stated.

    Parameters
    ----------

    df : dataframe
        Dataframe with the final features column

    final_feature_col : str
        column in dataframe that holds the final features

    shuffle : bool
        shuffle train and test datasets

    Returns
    -------
    X_train, X_test, y_train, y_test
        Split of the dataframe into train and test set

    """
    random.seed(42)

    x = df.get(final_features).to_numpy()
    labels = df.get("Targets").to_numpy()

    # prepare to concat
    concat_features = []
    for arr in x:
        concat_features.append((np.concatenate(arr)))

    train_idx = df.index[df["Train"] == 1].tolist()

    X_train = np.array([concat_features[i] for i in train_idx])
    y_train = np.array([labels[i] for i in train_idx])

    # X_test, y_test
    test_idx = df.index[df["Train"] == 0].tolist()

    X_test = np.array([concat_features[i] for i in test_idx])
    y_test = np.array([labels[i] for i in test_idx])

    if shuffle is True:
        train_data = list(zip(X_train, y_train))
        random.shuffle(train_data)
        X_train, y_train = zip(*train_data)
        test_data = list(zip(X_test, y_test))
        random.shuffle(test_data)
        X_test, y_test = zip(*test_data)

        # convert back to numpy array
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test


def print_metric(clf, y_test, y_pred):
    """[Generates a dictionary of metrics for a given classifier ]

    Args:
    classifier ([type]): [machine learning classifier]
    y_test ([np.array]): [true labels]
    y_pred ([np.array]): [predicted labels]

    Returns:
    [type]: [description]
    """
    results = {}

    # Generate sensitivity, specificity metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    specificity = (tn / (tn + fp)) * 100
    sensitivity = (tp / (tp + fn)) * 100

    # Generate accuracy
    acc = accuracy_score(y_test, y_pred) * 100

    # Generate MCC
    mcc = matthews_corrcoef(y_test, y_pred)

    # Generate ROC AUC Score
    roc_auc = roc_auc_score(y_test, y_pred)

    results["sensitivity"] = sensitivity
    results["specificity"] = specificity
    results["acc"] = acc
    results["mcc"] = mcc
    results["roc_auc"] = roc_auc
    return results


def internal_validation(clf, X_train, y_train, cv=5):
    results = {}

    # Internal Validation using 5-fold CV
    score = cross_val_score(clf, X_train, y_train, scoring="accuracy", cv=cv, n_jobs=-1)
    internal_cv_acc = score.mean() * 100
    internal_cv_acc_std = score.std() * 100

    results["internal_acc"] = internal_cv_acc
    results["internal_cv_acc_std"] = internal_cv_acc_std

    return results


def read_pickle(file_path, length=None):
    df = pd.read_pickle(file_path)
    df = df[df["Length"] >= length].reset_index(drop=True)
    return df


def main(dataname = None, scaling=True, thresholding_models=False, window = None):
    scale = scaling
    
    if dataname == "A":
        
        df = read_pickle(
        
        # Dataset A (alternate dataset)
        "ACP-alternate_dataset_preprocessed_window_" + str(window) +".pkl", length = window
        )  # insert model pickle file here
        final_features = [
            "Mean_magnus",
            "Sum_magnus",
            #"Natural_Vector",
            #"L0_ev_avg",
            #"L1_ev_avg",
            #"L0_ev_count",
            #"L1_ev_count",
            #"N15C15_natural", 
            #"N15C15_magnus_mean"
            ]
    elif dataname == "B":
        df = read_pickle(
        # Dataset B (main dataset) 
        "ACP-main_dataset_preprocessed_window_" + str(window) +".pkl", length = window
        )  # insert model pickle file here
        
        # features for main datset (Dataset B)
        
        final_features = [
            "Mean_magnus",
            "Sum_magnus",
    #        "Natural_Vector",
            #"L0_ev_avg",
            #"L1_ev_avg",
       #     "L0_ev_count",
        #    "L1_ev_count",
        #    "N15C15_natural", 
        #    "N15C15_magnus_mean"
            ]
    else:
        raise ValueError(f"Invalid dataset name {dataname}. Choose 'A' for alternate dataset or 'B' for main dataset.")


    # df = read_pickle('ACP-preprocessed-ßinal_v2.pkl')

    print(f"Dataframe shape:{df.shape}")
    print(f"Scaling is {scale}.")

    X_train, X_test, y_train, y_test = train_test_split(
        df, final_features=final_features, shuffle=True
    )

    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {X_test.shape}")

    if scaling is True:
        scaler = preprocessing.StandardScaler()
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    # training loop
    all_results = [] 
    clfs = []
    best_acc = 0

    for i in range(iters):
        clf = ExtraTreesClassifier(**params)
        #clf = RandomForestClassifier(**params)
        #clf = GradientBoostingClassifier(**params)
        
        dic1 = internal_validation(clf, X_train, y_train)  # internal validation result
        clf.fit(X_train, y_train)
        clfs.append(clf)
        y_pred = clf.predict(X_test)
        dic2 = print_metric(clf, y_test, y_pred) # test performance
        dic1.update(dic2) # update the table with test performance
        all_results.append(dic1)  # test performance
        print(f"{i+1}/{iters} done!")

        if dic1["acc"] >= best_acc:
            best_acc = dic1["acc"]

    results_df = pd.DataFrame(all_results) # test performance
    results_df = results_df.round(3)

    # get a dataframe of the parameters
    params["iters"] = iters
    params["scale"] = scale
    features_list = pd.DataFrame(final_features)
    hparams_list = pd.DataFrame.from_dict(params, orient="index")
    utils_df = pd.concat([features_list, hparams_list], axis=1)

    # get a dataframe of the mean results
    mean_results_df = results_df.mean() # mean test performance
    mean_results_df = mean_results_df.transpose()
    median_results_df = results_df.median() # mean test performance
    median_results_df = median_results_df.transpose() # mean test performance

    #print(mean_results_df)
    print(median_results_df)

    # Calculate best threshold and results

    if thresholding_models is True:

        best_metrics_all = []
        counter = 0
        for classifier in clfs:
            best_metrics_clf = None
            best_acc = 0
            best_threshold = 0
            preds = classifier.predict_proba(X_test)[:, 1]

            fpr, tpr, thresholds = roc_curve(y_test, preds)

            for j in range(1, len(thresholds)):
                if 0.40 <= thresholds[j] <= 0.60:  # probability threshold tuning
                    y_pred = (
                        classifier.predict_proba(X_test)[:, 1] >= thresholds[j]
                    ).astype(int)
                    metrics = print_metric(classifier, y_test, y_pred)

                    if metrics["acc"] >= best_acc:
                        best_acc = metrics["acc"]
                        best_metrics_clf = metrics
                        best_threshold = thresholds[j]

            best_metrics_clf["threshold"] = best_threshold

            best_metrics_all.append(best_metrics_clf)

            counter += 1
            print(f" Done {counter}/{iters}")
        print("Done finding best thresholded models!")

        best_metrics_all_df = pd.DataFrame(best_metrics_all)
        
        
        # Calculate mean and standard deviation for each column

        mean_values = best_metrics_all_df.mean()
        median_values = best_metrics_all_df.median()
        std_dev_values = best_metrics_all_df.std()

        # Create a DataFrame with these values
        summary_df = pd.DataFrame([mean_values, median_values, std_dev_values], index=['Mean', 'Median', 'Standard Deviation'])

        # Append the summary DataFrame to the original DataFrame
        best_metrics_all_df = pd.concat([best_metrics_all_df, summary_df])

    # setting file path
    filePath = "Etrees"
    time = getStandardTime()
    os.makedirs(filePath, exist_ok=True)

    # read in hyperparameters as well
    with pd.ExcelWriter(f"{filePath}/output_{time}.xlsx") as writer:

        mean_results_df.to_excel(writer, sheet_name="mean_results")  # mean test performance using prediction threshold of 0.5
        median_results_df.to_excel(writer, sheet_name="median_results")  # mean test performance using prediction threshold of 0.5
        results_df.to_excel(writer, sheet_name="results")   # test performance using prediction threshold of 0.5
        utils_df.to_excel(writer, sheet_name="parameters") # model parameters

        if thresholding_models is True:
            best_metrics_all_df.to_excel(writer, sheet_name="Thresholded_results")
            best_threshold_result =  best_metrics_all_df.loc[ best_metrics_all_df["acc"] == max(best_metrics_all_df["acc"])]
            best_threshold_result.to_excel(writer, sheet_name="Best_thresholded_results")

            print(
                best_metrics_all_df.loc[
                    best_metrics_all_df["acc"] == max(best_metrics_all_df["acc"])
                ]
            ) # test performance using optimal prediction threshold
            print(summary_df)
        print("Results have been populated to excel!")


if __name__ == "__main__":
    main(dataname = "B", scaling=True, thresholding_models=False, window = 4)

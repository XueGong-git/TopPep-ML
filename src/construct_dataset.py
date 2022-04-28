import numpy as np
import pandas as pd

import utils


def read_input(path):
    f = open(path, "r")
    lst = []
    for line in f:
        lst.append(line.rstrip())
    return lst


def loading_raw_dataset():
    # reading inputs
    neg_train = read_input("data/alternate_dataset/neg_train_alternate.txt")
    pos_train = read_input("data/alternate_dataset/pos_train_alternate.txt")

    neg_test = read_input("data/alternate_dataset/neg_test_alternate.txt")
    pos_test = read_input("data/alternate_dataset/pos_test_alternate.txt")

    # creating train labels
    pos_train_labels = np.ones(len(pos_train))
    neg_train_labels = np.zeros(len(neg_train))

    # creating test labels
    pos_test_labels = np.ones(len(pos_test))
    neg_test_labels = np.zeros(len(neg_test))

    # joining pos_train and neg_train
    X_train = pos_train + neg_train
    y_train = np.concatenate((pos_train_labels, neg_train_labels), axis=None)
    y_train = y_train.astype(int)

    X_test = pos_test + neg_test
    y_test = np.concatenate((pos_test_labels, neg_test_labels), axis=None)
    y_test = y_test.astype(int)

    data = {"Peptides": X_train, "Targets": y_train}

    df = pd.DataFrame(data)

    # shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)

    # add a column to indicate training set
    df["Train"] = 1

    df_test = pd.DataFrame({"Peptides": X_test, "Targets": y_test})

    # Assign 0 to test set rows
    df_test["Train"] = 0

    df = df.append(df_test).reset_index(drop=True)

    # change col type of targets and train to categorical
    df["Targets"] = df["Targets"].astype("category")
    df["Train"] = df["Train"].astype("category")

    return df


def preprocessing_loaded_data(df):
    # calculate length of each sequence
    df["Length"] = df.apply(lambda x: len(x["Peptides"]), axis=1)
    min_size = 5
    df = df[df["Length"] >= min_size].copy()
    return df


def SpectPep_encoding(df):
    df["B1_count"] = df.apply(lambda x: utils.B1(x["Peptides"]), axis=1)
    df["B1_avg"] = df.apply(lambda x: utils.B1(x["Peptides"], feature="avg"), axis=1)
    df["B1_std"] = df.apply(lambda x: utils.B1(x["Peptides"], feature="std"), axis=1)

    # Generate normalised laplacian L0
    df["L0_count"] = df.apply(lambda x: utils.L0(x["B1_count"]), axis=1)
    df["L0_avg"] = df.apply(lambda x: utils.L0(x["B1_avg"]), axis=1)
    df["L0_std"] = df.apply(lambda x: utils.L0(x["B1_std"]), axis=1)

    # Generate eigenvalues from L0
    df["L0_ev_count"] = df.apply(lambda x: utils.gen_eigenvalues(x["L0_count"]), axis=1)
    df["L0_ev_avg"] = df.apply(lambda x: utils.gen_eigenvalues(x["L0_avg"]), axis=1)
    df["L0_ev_std"] = df.apply(lambda x: utils.gen_eigenvalues(x["L0_std"]), axis=1)

    # Generate L1 matrix
    df["L1_count"] = df.apply(lambda x: utils.L1(x["Peptides"]), axis=1)
    df["L1_avg"] = df.apply(lambda x: utils.L1(x["Peptides"], feature="avg"), axis=1)
    df["L1_std"] = df.apply(lambda x: utils.L1(x["Peptides"], feature="std"), axis=1)

    # Generate L1 eigenvalues
    df["L1_ev_count"] = df.apply(lambda x: utils.gen_eigenvalues(x["L1_count"]), axis=1)
    df["L1_ev_avg"] = df.apply(lambda x: utils.gen_eigenvalues(x["L1_avg"]), axis=1)
    df["L1_ev_std"] = df.apply(lambda x: utils.gen_eigenvalues(x["L1_std"]), axis=1)
    return df.drop(
        columns=[
            "B1_count",
            "B1_avg",
            "B1_std",
            "L0_count",
            "L0_avg",
            "L0_std",
            "L1_count",
            "L1_avg",
            "L1_std",
        ],
        inplace=True,
    )


def magnus_encoding(df):
    df["Mean_magnus"] = df.apply(
        lambda x: utils.agg_magnus_vector(x["Peptides"], 5, aggregation="mean"), axis=1
    )
    df["Sum_magnus"] = df.apply(
        lambda x: utils.agg_magnus_vector(x["Peptides"], 5, aggregation="sum"), axis=1
    )

    return df


def natural_encoding(df):
    df["Natural_Vector"] = df.apply(
        lambda x: utils.natural_vector(x["Peptides"]), axis=1
    )

    return df


def main():
    df = loading_raw_dataset()
    df = preprocessing_loaded_data(df)
    # df = SpectPep_encoding(df)      # takes about 10mins to run
    df = magnus_encoding(df)
    df = natural_encoding(df)
    print(df.head())
    print(df.shape)

    # save dataframe as a pickle file
    df.to_pickle("ACP-alternate_dataset_preprocessed_v1.pkl", protocol=4)


if __name__ == "__main__":
    main()

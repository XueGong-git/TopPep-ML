import numpy as np
import pandas as pd

import utils


def read_input(path):
    f = open(path, "r")
    lst = []
    for line in f:
        lst.append(line.rstrip())
    return lst


def loading_raw_dataset(dataname) :
    # reading inputs
    
    if dataname == "A":
        
        # alternate dataset (dataset A)
        neg_train = read_input("../data/alternate_dataset/neg_train_alternate.txt")
        pos_train = read_input("../data/alternate_dataset/pos_train_alternate.txt")
    
        neg_test = read_input("../data/alternate_dataset/neg_test_alternate.txt")
        pos_test = read_input("../data/alternate_dataset/pos_test_alternate.txt")
        
    elif dataname == "B":
        # main dataset (dataset B)
        neg_train = read_input("../data/main_dataset/neg_train_main.txt")
        pos_train = read_input("../data/main_dataset/pos_train_main.txt")
    
        neg_test = read_input("../data/main_dataset/neg_test_main.txt")
        pos_test = read_input("../data/main_dataset/pos_test_main.txt")
        
    elif dataname == "mACPpred2":
        neg_train = read_input("../data/mACPpred2.0/neg_train_mACPpred2.txt")
        pos_train = read_input("../data/mACPpred2.0/pos_train_mACPpred2.txt")
    
        neg_test = read_input("../data/mACPpred2.0/neg_test_mACPpred2.txt")
        pos_test = read_input("../data/mACPpred2.0/pos_test_mACPpred2.txt")
        
    else:
        raise ValueError(f"Invalid dataset name {dataname}. Choose 'A' for alternate dataset or 'B' for main dataset.")
    
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

    pd.concat([df, df_test], ignore_index=True)
    #df = df.append(df_test).reset_index(drop=True)
    df = pd.concat([df, df_test], ignore_index=True)

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
    df = df.drop(
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
        ]
        , inplace=False,
    )
    return df


def magnus_encoding(df, window):
    df["Mean_magnus"] = df.apply(
        lambda x: utils.agg_magnus_vector(x["Peptides"], step_size = window, aggregation="mean"), axis=1
    )
    df["Sum_magnus"] = df.apply(
        lambda x: utils.agg_magnus_vector(x["Peptides"], step_size = window, aggregation="sum"), axis=1
    )
    




    # N5
    df["N5_magnus_mean"] = df.apply(
    lambda x: utils.agg_magnus_vector(x["N5"], step_size = window, aggregation="mean"), axis=1
    )
    
    # N10
    df["N10_magnus_mean"] = df.apply(
    lambda x: utils.agg_magnus_vector(x["N10"], step_size = window, aggregation="mean"), axis=1
    )
    
    # N15
    df["N15_magnus_mean"] = df.apply(
    lambda x: utils.agg_magnus_vector(x["N15"], step_size = window, aggregation="mean"), axis=1
    )
    
    # C5
    df["C5_magnus_mean"] = df.apply(
    lambda x: utils.agg_magnus_vector(x["C5"], step_size = window, aggregation="mean"), axis=1
    )
    
    # C10
    df["C10_magnus_mean"] = df.apply(
    lambda x: utils.agg_magnus_vector(x["C10"], step_size = window, aggregation="mean"), axis=1
    )
    
    # C15 terminus magnus
    df["C15_magnus_mean"] = df.apply(
    lambda x: utils.agg_magnus_vector(x["C15"], step_size = window, aggregation="mean"), axis=1
    )
    
    # N5C5 terminus magnus
    df["N5C5_magnus_mean"] = df.apply(
    lambda x: utils.agg_magnus_vector(x["N5C5"], step_size = window, aggregation="mean"), axis=1
    )
    
    # N10C10 terminus magnus
    df["N10C10_magnus_mean"] = df.apply(
      lambda x: utils.agg_magnus_vector(x["N10C10"], step_size = window, aggregation="mean"), axis=1
      )
    
    # N15C15 terminus magnus
    df["N15C15_magnus_mean"] = df.apply(
    lambda x: utils.agg_magnus_vector(x["N15C15"], step_size = window, aggregation="mean"), axis=1
    )
    
    
    return df


def natural_encoding(df):
    df["Natural_Vector"] = df.apply(
        lambda x: utils.natural_vector(x["Peptides"]), axis=1
    )
    df["N5_natural"] = df.apply(
        lambda x: utils.natural_vector(x["N5"]), axis=1
    )
    df["N10_natural"] = df.apply(
        lambda x: utils.natural_vector(x["N10"]), axis=1
    )
    df["N15_natural"] = df.apply(
        lambda x: utils.natural_vector(x["N15"]), axis=1
    )
    df["C5_natural"] = df.apply(
        lambda x: utils.natural_vector(x["C5"]), axis=1
    )
    df["C10_natural"] = df.apply(
        lambda x: utils.natural_vector(x["C10"]), axis=1
    )
    df["C15_natural"] = df.apply(
        lambda x: utils.natural_vector(x["C15"]), axis=1
    )
    df["N5C5_natural"] = df.apply(
        lambda x: utils.natural_vector(x["N5C5"]), axis=1
    )
    df["N10C10_natural"] = df.apply(
        lambda x: utils.natural_vector(x["N10C10"]), axis=1
    )
    df["N15C15_natural"] = df.apply(
        lambda x: utils.natural_vector(x["N15C15"]), axis=1
    )

    return df


def terminal_composition_raw(df):
    df["N5"] = df.apply(
        lambda x: utils.terminal_composition_construct(x["Peptides"], "N", x=5), axis=1
    )
    df["N10"] = df.apply(
        lambda x: utils.terminal_composition_construct(x["Peptides"], "N", x=10), axis=1
    )
    df["N15"] = df.apply(
        lambda x: utils.terminal_composition_construct(x["Peptides"], "N", x=15), axis=1
    )
    df["C5"] = df.apply(
        lambda x: utils.terminal_composition_construct(x["Peptides"], "C", y=5), axis=1
    )
    df["C10"] = df.apply(
        lambda x: utils.terminal_composition_construct(x["Peptides"], "C", y=10), axis=1
    )
    df["C15"] = df.apply(
        lambda x: utils.terminal_composition_construct(x["Peptides"], "N", y=15), axis=1
    )
    df["N5C5"] = df.apply(
        lambda x: utils.terminal_composition_construct(x["Peptides"], "NC", x=5, y=5),
        axis=1,
    )
    df["N10C10"] = df.apply(
        lambda x: utils.terminal_composition_construct(x["Peptides"], "NC", x=10, y=10),
        axis=1,
    )
    df["N15C15"] = df.apply(
        lambda x: utils.terminal_composition_construct(x["Peptides"], "NC", x=15, y=15),
        axis=1,
    )
    return df

def read_pickle(file_path, length=5):
    df = pd.read_pickle(file_path)
    df = df[df["Length"] >= length].reset_index(drop=True)

    return df


def main(window):

    
    # dataset A for alternate dataset
    df = loading_raw_dataset(dataname="A")
    df = preprocessing_loaded_data(df)
    df = SpectPep_encoding(df)         # takes about 10mins to run
    df = terminal_composition_raw(df)
    df = magnus_encoding(df, window)
    df = natural_encoding(df)


    # save dataframe as a pickle file
    df.to_pickle("ACP-alternate_dataset_preprocessed_window_" + str(window) +".pkl", protocol=4)

    # dataset B for main dataset
    df = loading_raw_dataset(dataname = "B")
    df = preprocessing_loaded_data(df)
    df = SpectPep_encoding(df)         # takes about 10mins to run
    df = terminal_composition_raw(df)
    df = magnus_encoding(df, window)
    df = natural_encoding(df)
    # save dataframe as a pickle file
    df.to_pickle("ACP-main_dataset_preprocessed_window_" + str(window) +".pkl", protocol=4)
    
    # dataset A for alternate dataset
    df = loading_raw_dataset(dataname="mACPpred2")
    df = preprocessing_loaded_data(df)
    df = SpectPep_encoding(df)         # takes about 10mins to run
    df = terminal_composition_raw(df)
    df = magnus_encoding(df, window)
    df = natural_encoding(df)


    # save dataframe as a pickle file
    df.to_pickle("mACPpred2_preprocessed_window_" + str(window) +".pkl", protocol=4)


if __name__ == "__main__":
    main(window = 5)

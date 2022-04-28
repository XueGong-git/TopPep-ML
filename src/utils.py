import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Setting up relevant hashmaps

std = list("ACDEFGHIKLMNPQRSTVWY")

pairwise = ["".join([i, j]) for i in std for j in std]

triplets = ["".join([i, j, k]) for i in std for j in std for k in std]

letter_dict = {v: k for k, v in enumerate(std)}
pair_combi_dict = {v: k for k, v in enumerate(pairwise)}  # ('AA', 0), ('AC', 1) ...
triplets_combi_dict = {
    v: k for k, v in enumerate(triplets)
}  # i.e. ("AAA": 0), ("AAC": 1), ("AAD": 2)...

lst = std + pairwise
magnus_dic = {v: k for k, v in enumerate(lst)}


# create B1 matrix
def B1(seq, feature=None, skip=0):
    """
    Takes in a sequence and generate a B1 boundary matrix

    Parameters
    ----------
    seq : str
        Sequence of peptide

    feature : str
        Define aggregation method to use to generate boundary matrix

    skip : int
        Number of tokens to skip for window size

    Returns
    -------
    array
        B1 boundary matrix representing the sequence

    """

    if feature == None:  # Case 1 : counting residues
        mat = np.zeros((20, 400))
        for i in range(len(seq) - 1 - skip):
            check = seq[i : i + 2 + skip]  # get subset of letters
            check = seq[0] + seq[-1]  # get first and last letter in subset
            if check in pairwise:
                row1 = letter_dict[
                    check[0]
                ]  # return corresponding index(row) in letter_dict
                row2 = letter_dict[check[1]]

                col = pair_combi_dict[
                    check
                ]  # return corresponding index(column) in combi_dict
                mat[row1][col] += 1
                mat[row2][col] += 1

    if feature == "avg" or feature == "std" or feature == "min" or feature == "max":
        dict_val = {k: [] for k in std}

        for val, letter in enumerate(seq, start=1):  # val starts from 1
            dict_val[letter].append(val)

        if feature == "avg":  # Case 2: Average Index of residues
            output = {k: np.mean(v) if len(v) > 0 else 0 for k, v in dict_val.items()}

        if feature == "std":  # Case 3: Standard deviation of residues
            output = {k: np.std(v) if len(v) > 1 else 0 for k, v in dict_val.items()}

        if feature == "min":  # Case 4: Min index of letter
            output = {k: min(v) if len(v) > 0 else 0 for k, v in dict_val.items()}

        if feature == "max":  # Case 5: Max index of letter
            output = {k: max(v) if len(v) > 0 else 0 for k, v in dict_val.items()}

        mat = np.zeros((20, 400))
        for i in range(len(seq) - 1):
            check = seq[i : i + 2]  # get a pair of adjacent letters
            if check in pairwise:
                row1 = letter_dict[
                    check[0]
                ]  # return corresponding index(row) in letter_dict
                row2 = letter_dict[check[1]]

                col = pair_combi_dict[
                    check
                ]  # return corresponding index(column) in combi_dict
                mat[row1][col] += output[check[0]]
                mat[row2][col] += output[check[1]]

    return mat  # (20,20)


def L0(mat, inv=False):
    """
    Takes in a B1 boundary matrix and applies Laplacian normalised
    transformations to generate a L0 Laplacian matrix.

    Parameters
    ----------

    mat : matrix
        B1 boundary matrix

    inv : bool
        Returns a (400,400) matrix since L0 is calculated in an inverse manner

    Returns
    -------

    matrix
        L0 matrix of size (20,20) if inv = False

    """
    if inv == False:
        L0_ = mat @ mat.transpose()  # return (20,20) matrix
        L_sym = laplacian_normalised(
            L0_, size=20
        )  # calculate laplacian normalised matrix
        return L_sym

    else:
        L0_inv = mat.transpose() @ mat  # return (400,400) matrix
        return L0_inv


def L1(seq, feature=None, skip=0):
    """
    Generates a laplacian normalised L1 matrix based on a given sequence

    Paramters
    ---------

    seq : str
        Sequence of peptide

    feature : str
        Define aggregation method to use to generate boundary matrix

    skip : int
        Number of tokens to skip for window size


    Returns
    -------
    matrix
        Laplacian normalised matrix of size (400,400)

    """

    if feature == None:
        mat = np.zeros((400, 8000))
        for i in range(len(seq) - 2 - skip):
            check = seq[i : i + 3]  # generate a subset
            lst = [
                check[:2],
                check[1:],
            ]  # generate consecutive 2-combinations, e.g ["AC", "AD", "CD"]

            col = triplets_combi_dict[check]

            for j in range(len(lst)):  # for each 2-combination, find corresponding row
                row = pair_combi_dict[lst[j]]
                mat[row][col] += 1

    if feature == "avg" or feature == "std" or feature == "min" or feature == "max":
        # create a dic to store the values (list of indexes of each pair)
        dict_val = {k: [] for k in pairwise}

        for val, i in enumerate(
            range(len(seq) - 1), start=1
        ):  # start index count from 1
            pair = seq[i : i + 2]

            dict_val[pair].append(val)

        if feature == "avg":
            output = {k: np.mean(v) if len(v) > 0 else 0 for k, v in dict_val.items()}

        if feature == "std":
            output = {k: np.std(v) if len(v) > 0 else 0 for k, v in dict_val.items()}

        if feature == "min":  # Case 4: Min index of letter
            output = {k: min(v) if len(v) > 1 else 0 for k, v in dict_val.items()}

        if feature == "max":  # Case 5: Max index of letter
            output = {k: max(v) if len(v) > 1 else 0 for k, v in dict_val.items()}

        mat = np.zeros((400, 8000))

        for i in range(len(seq) - 2):
            check = seq[i : i + 3]  # take a triplet, e.g. "ACD"
            lst = [
                check[0] + check[1],
                check[1] + check[2],
            ]  # generate 2 2-combinations, e.g ["AC", "AD", "CD"]
            col = triplets_combi_dict[check]

            for j in lst:  # for each 2-combination, find corresponding row
                row = pair_combi_dict[j]
                mat[row][col] += output[j]

    L1 = mat @ mat.transpose() + L0(
        B1(seq, feature, skip), inv=True
    )  # apply L1 formula to produce (400,400) matrix

    L1_sym = laplacian_normalised(L1, size=400)

    return L1_sym


# calculate Laplacian Normalised Matrix given a matrix as input
def laplacian_normalised(matrix, size: int):
    """
    For a given boundary matrix, apply transformations to obtain
    laplacian normalised matrix.

    Parameters
    ----------
    matrix : matrix
        Boundary matrix

    size : int
        Define size of output

    Returns
    -------
    matrix
        Laplacian normalised matrix of shape (size, size)


    """

    diags = np.diagonal(matrix)  # store diagonals

    off_diags = (
        np.sum(matrix, axis=1) - diags
    )  # calculate sum of off-diagonals to be new diagonal value

    matrix = -matrix  # every value except diagonals to be negative

    matrix[
        range(size), range(size)
    ] = off_diags  # diagonals to be the positive row-wise sum of all the off-diagonals

    D = np.zeros([size, size])

    with np.errstate(divide="ignore"):
        off_diags = off_diags**-0.5  # need to apply -.5 power to the diagonals
        off_diags = np.where(off_diags == np.inf, 0, off_diags)  # change all np.inf = 0

    D[range(size), range(size)] = off_diags

    L_sym = D @ matrix @ D

    return L_sym


def gen_eigenvalues(input):
    """
    Generate eigenvalues given a matrix

    Parameters
    ----------
    input : matrix
        Boundary matrix

    Returns
    -------

    array
        array representing the eigenvalues generated

    """
    vals = np.linalg.eigvalsh(input)
    return np.array(
        [i if i > 1e-3 else 0 for i in vals]
    )  # for eigenvalues less than 1e-3, fix them = 0


def k_mer_magnus(seq, length=2):
    """
    Takes in an input k-mer sequence and returns a magnus vector

    Parameters
    ----------

    seq : str
        Input sequence of specified window size

    length : int
        Specify maximum length of subsequences

    Returns
    -------
    array
        k-mer magnus vector representing the sequence

    """

    output = [[]]

    # fmt: off
    # generate all subsequences from given sequence iteratively up to specified length
    for i in seq:
        output += [j + [i] for j in output if len(j) <= length - 1]
    # fmt: on

    subsequences = ["".join(i) for i in output]  # e.g. ['F', 'F', 'FF', 'C',...]

    subsequences = subsequences[1:]  # remove first item in list which is empty

    output_tokens = [
        magnus_dic[i] for i in subsequences
    ]  # map subsequences to index e.g. [5, 5, 105, 2...]

    arr = np.zeros(
        420
    )  # since k-mer magnus vector only represents combinations up to 2-mer, i.e. 20 + 20^2 = 420

    # fmt: off
    for val in output_tokens:  # for each subsequence, add one to the correspond index in arr
        arr[val] += 1
    return arr
    # fmt: on


def agg_magnus_vector(seq, step_size=5, aggregation=None):
    """
    Takes in an full length input sequence, generates a k-mer magnus vector for each window size and returns an aggregated magnus vector

    Parameters
    ----------

    seq : str
        Full input sequence

    step_size : int
        Specify window size for non-overlapping k-mers in seq

    aggregation: str
        Define aggregation method to use to generate final aggregate magnus vector

    Returns
    -------
    array
        Aggregated magnus vector representing the sequence of size (420)

    """
    magnus = []
    no_kmers = len(seq) // step_size

    for i in range(0, len(seq), step_size):
        if (
            len(seq) - i >= step_size
        ):  # drop the excess parts of the sequence towards the end if it is smaller than the step size
            magnus.append(
                k_mer_magnus(seq[i : i + step_size])
            )  # for every k-mer, call k_mer_magnus function to generate magnus vector and store it

    # Aggregate magnus vectors based on aggregation method

    if aggregation == "mean":
        output = (
            np.array(magnus).sum(axis=0) / no_kmers
        )  # sum up all magnus vectors and divide it by # of kmers

    elif aggregation == "sum":
        output = np.array(magnus).sum(axis=0)

    elif aggregation == "std":
        output = np.array(magnus).std(axis=0)

    return output


def natural_vector(peptide):

    vec = []
    for amino in std:
        test_amino = amino
        counter_letter = 0
        counter_total_index = 0
        counter_scaled_2nd_moment = 0

        for idx, i in enumerate(peptide, 1):
            if i == test_amino:
                counter_letter += 1
                counter_total_index += idx

        if counter_letter > 0:
            mean_position = counter_total_index / counter_letter
        else:
            mean_position = 0

        if counter_letter >= 2:
            for idx, i in enumerate(peptide, 1):
                if i == test_amino:
                    counter_scaled_2nd_moment = (idx - mean_position) ** 2

            scaled_2nd_moment = counter_scaled_2nd_moment / (
                counter_letter * len(peptide)
            )
        else:
            scaled_2nd_moment = 0

        vec.append([counter_letter, mean_position, scaled_2nd_moment])

    return np.array(vec).reshape(-1)

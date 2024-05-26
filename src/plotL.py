# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:57:40 2024

@author: jie14
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import utils

import utils
import construct_dataset

seq = "ACDE"
feature = "avg"


std = list("ACDE")
len_std = len(std)
pairwise = ["".join([i, j]) for i in std for j in std]

triplets = ["".join([i, j, k]) for i in std for j in std for k in std]

letter_dict = {v: k for k, v in enumerate(std)}
pair_combi_dict = {v: k for k, v in enumerate(pairwise)}  # ('AA', 0), ('AC', 1) ...
triplets_combi_dict = {
    v: k for k, v in enumerate(triplets)
}  # i.e. ("AAA": 0), ("AAC": 1), ("AAD": 2)...

lst = std + pairwise
magnus_dic = {v: k for k, v in enumerate(lst)}



### compute boundary operator B1

dict_val = {k: [] for k in std}

for val, letter in enumerate(seq, start=1):  # val starts from 1
    dict_val[letter].append(val)

output = {k: np.mean(v) if len(v) > 0 else 0 for k, v in dict_val.items()}


mat = np.zeros((len_std, len_std*len_std))
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





# Generate normalised laplacian L0
L0 = mat @ mat.transpose()  # return (20,20) matrix


### Process L0
diags = np.diagonal(L0)  # store diagonals

off_diags = (
    np.sum(L0, axis=1) - diags
)  # calculate sum of off-diagonals to be new diagonal value

L0_processed = -L0  # every value except diagonals to be negative
size = len(L0)
L0_processed[
    range(size), range(size)
] = off_diags  # diagonals to be the positive row-wise sum of all the off-diagonals

L_sym = laplacian_normalised(
     L0_, size=20
 )  # calculate laplacian normalised matrix

sns.heatmap(L0_norm, cmap='bwr', cbar=True, center=0, linecolor='black', linewidth=0.5, annot=True, fmt=".2f")
labels = std
tick_positions = np.arange(len(labels)) + 0.5  # Adjusted positions for centered labels
plt.xticks(ticks=tick_positions, labels=labels)
plt.yticks(ticks=tick_positions, labels=labels)
plt.gca().set_aspect('equal', adjustable='box')
plt.tick_params(axis='both', which='both', length=0)

# Draw the outline
ax = plt.gca()
ax.add_patch(plt.Rectangle((0, 0), len(labels), len(labels), fill=False, edgecolor='black', linewidth=2))

plt.savefig('L0_norm.png', dpi=300, bbox_inches='tight')
plt.show()




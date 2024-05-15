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
import construct_dataset

Peptides = "RPRCWIKIKFRCKSLKF"
df = construct_dataset.loading_raw_dataset(dataname = "B")

B1_count = utils.B1(Peptides)
B1_avg = utils.B1(Peptides, feature="avg")
B1_std = utils.B1(Peptides, feature="std")

# Generate normalised laplacian L0
L0_count =  utils.L0(B1_count)
L0_avg =  utils.L0(B1_avg)
L0_std = utils.L0(B1_std)






sns.heatmap(L0_count, cmap='bwr', vmin=-1, vmax=1)
labels = list("ACDEFGHIKLMNPQRSTVWY")
plt.xticks(ticks=np.arange(len(labels)), labels=labels)
plt.yticks(ticks=np.arange(len(labels)), labels=labels)
plt.gca().set_aspect('equal', adjustable='box')
# Draw the outline
ax = plt.gca()

for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

plt.show()

sns.heatmap(L0_avg, cmap='bwr', vmin=-1, vmax=1)
plt.xticks(ticks=np.arange(len(labels)), labels=labels)
plt.yticks(ticks=np.arange(len(labels)), labels=labels)
plt.gca().set_aspect('equal', adjustable='box')
# Draw the outline
ax = plt.gca()

for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

plt.show()
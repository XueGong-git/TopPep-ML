#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspecting column sizes in a .pkl dataset
"""
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample Data
import pandas as pd
import numpy as np

from pprint import pprint

# Define the window size
window = 5

# Path to your .pkl file
file_path = f"ACP-alternate_dataset_preprocessed_window_{window}.pkl"

# Load the .pkl file
with open(file_path, "rb") as f:
    data = pickle.load(f)

# Display the contents briefly
print("Contents of the .pkl file (first 2 entries, if iterable):")
if isinstance(data, dict) or isinstance(data, list):
    pprint(data if len(data) <= 2 else data[:2])  # Show only the first 2 entries

print("\nNumber of elements in each column:")
column_sizes = data.apply(lambda col: col.count())  # Count non-null elements in each column
for col_name, col_size in column_sizes.items():
    print(f"Column: {col_name}, Number of Entries: {col_size}")
    

#for i in range(1936):
#    if i in data.index:
#        try:
#            correlation = np.corrcoef(data['L0_ev_count'][i], data['L0_ev_avg'][i])[0, 1]
#           print(correlation)
#        except Exception as e:
#            print(f"Error at row {i}: {e}")
#    else:
#        print(f"Row {i} does not exist. Skipping.")


# Load the .npy file
X_test = np.load("X_test.npy")
Y_test = np.load("Y_test.npy")


# Convert the NumPy array to a Pandas DataFrame
# If the data is a 2D array
X_test_df = pd.DataFrame(X_test)
Y_test_df = pd.DataFrame(Y_test)



# Display additional details
print(f"Shape of DataFrame: {X_test_df.shape}")
print(X_test_df.info())

feature_column = np.array(data['Mean_magnus'].tolist())

plot_data = pd.DataFrame({
    'Feature': feature_column[:, -1].tolist(),
    'Outcome': data['Targets'].tolist()
})


# Boxplot
sns.boxplot(x='Outcome', y='Feature', data=plot_data)
plt.title('Boxplot of Feature by Outcome')
plt.show()



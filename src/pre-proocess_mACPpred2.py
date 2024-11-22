#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:55:33 2024

@author: gongxue
"""

# Read the FASTA file and write sequences to a text file
# Open the original FASTA file and two new files for positive and negative sequences
with open("../data/mACPpred2.0/mACPpred2.0_training.fasta", "r") as fasta_file, \
     open("../data/mACPpred2.0/pos_train_mACPpred2.txt", "w") as positive_file, \
     open("../data/mACPpred2.0/neg_train_mACPpred2.txt", "w") as negative_file:
    
    # Variables to store the current header and sequence
    header = ""
    sequence = ""
    
    for line in fasta_file:
        line = line.strip()
        
        if line.startswith(">"):  # Header line
            # Write the previous sequence to the appropriate file
            if header and sequence:
                if "positive" in header.lower():
                    positive_file.write(f"{sequence}\n")
                elif "negative" in header.lower():
                    negative_file.write(f"{sequence}\n")
            
            # Update the header and reset sequence
            header = line
            sequence = ""
        else:
            # Concatenate sequence lines
            sequence += line
    
    # Write the last sequence to the appropriate file
    if header and sequence:
        if "positive" in header.lower():
            positive_file.write(f"{sequence}\n")
        elif "negative" in header.lower():
            negative_file.write(f"{sequence}\n")
            
            

with open("../data/mACPpred2.0/mACPpred2.0_training.fasta", "r") as fasta_file:

    num_lines = sum(1 for line in fasta_file)/2
    # Print the result
    print(f"Number of train: {num_lines}")
        
with open("../data/mACPpred2.0/pos_train_mACPpred2.txt", "r") as positive_file:

    num_lines = sum(1 for line in positive_file)
    # Print the result
    print(f"Number of pos train: {num_lines}")

with open("../data/mACPpred2.0/neg_train_mACPpred2.txt", "r") as negative_file:

    num_lines = sum(1 for line in negative_file)
    # Print the result
    print(f"Number of negtrain: {num_lines}")
            
# Read the FASTA file and write sequences to a text file
# Open the original FASTA file and two new files for positive and negative sequences
with open("../data/mACPpred2.0/mACPpred2.0_testing.fasta", "r") as fasta_file, \
     open("../data/mACPpred2.0/pos_test_mACPpred2.txt", "w") as positive_file, \
     open("../data/mACPpred2.0/neg_test_mACPpred2.txt", "w") as negative_file:
    
    # Variables to store the current header and sequence
    header = ""
    sequence = ""
    
    for line in fasta_file:
        line = line.strip()
        
        if line.startswith(">"):  # Header line
            # Write the previous sequence to the appropriate file
            if header and sequence:
                if "positive" in header.lower():
                    positive_file.write(f"{sequence}\n")
                elif "negative" in header.lower():
                    negative_file.write(f"{sequence}\n")
            
            # Update the header and reset sequence
            header = line
            sequence = ""
        else:
            # Concatenate sequence lines
            sequence += line
    
    # Write the last sequence to the appropriate file
    if header and sequence:
        if "positive" in header.lower():
            positive_file.write(f"{sequence}\n")
        elif "negative" in header.lower():
            negative_file.write(f"{sequence}\n")


with open("../data/mACPpred2.0/mACPpred2.0_testing.fasta", "r") as fasta_file:
    num_lines = sum(1 for line in fasta_file)/2
    # Print the result
    print(f"Number of test: {num_lines}")
            
with open("../data/mACPpred2.0/pos_test_mACPpred2.txt", "r") as positive_file:

    num_lines = sum(1 for line in positive_file)
    # Print the result
    print(f"Number of pos test: {num_lines}")

with open("../data/mACPpred2.0/neg_test_mACPpred2.txt", "r") as negative_file:

    num_lines = sum(1 for line in negative_file)
    # Print the result
    print(f"Number of neg test: {num_lines}")
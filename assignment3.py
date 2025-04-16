# Jessica Reyes
# Machine Learning Assignment 3

import pandas as pd

# reads csvs for g_data and n_data
g_data = pd.read_csv("g_data.csv")
n_data = pd.read_csv("n_data.csv")

# initialize column names with label data (type of protein) and string data (amino acids)
g_labels = g_data.columns[0] # g_data column a
g_strings = g_data.columns[3] # g_data column d
n_labels = n_data.columns[0] # n_data column a
n_strings = n_data.columns[3] # n_data column d

# create new column 'length' for both dataframes for easy manipulation
g_data['length'] = g_data[g_strings].astype(str).str.len()
n_data['length'] = n_data[n_strings].astype(str).str.len()

# using pandas functions to find answers to assignment 3 prompts for g_data
g_num_proteins = len(g_data) # number of proteins (rows) in g_data
g_num_labels = g_data[g_labels].nunique() # number of unique labels (1-4) in g_data
g_proteins_per_class = g_data[g_labels].value_counts().sort_index() # sorts number of amino acids per class by label in g_data (ascending)
g_avg_length_per_class = g_data.groupby(g_labels)['length'].mean() # finds avg amino acid length for g_data by label
g_min_max_per_class = g_data.groupby(g_labels)['length'].agg(['min', 'max']) # finds min and max amino acid lengths for g_data by label
g_formatting_help = g_proteins_per_class.to_string(index=True, header=False)

# using pandas functions to find answers to assignment 3 prompts for g_data
n_num_proteins = len(n_data) # number of proteins (rows) in n_data
n_num_labels = n_data[n_labels].nunique() # number of unique labels (1-4) in n_data
n_proteins_per_class = n_data[n_labels].value_counts().sort_index() # sorts number of amino acids per class by label in n_data (ascending)
n_avg_length_per_class = n_data.groupby(n_labels)['length'].mean() # finds avg amino acid length for n_data by label
n_min_max_per_class = n_data.groupby(n_labels)['length'].agg(['min', 'max']) # finds min and max amino acid lengths for n_data by label
n_formatting_help = n_proteins_per_class.to_string(index=True, header=False)

# needed to add this to eliminate a formatting error
g_min_max_per_class.index.name = None
n_min_max_per_class.index.name = None

# prints answers to assignment 3 for g_data and n_data
print("GRAM-POSITIVE BACTERIA")
print(f"Number of proteins: {g_num_proteins} \n")
print(f"Number of Labels: {g_num_labels} \n")
print(f"Number of proteins in each class: \n{g_formatting_help} \n")
print(f"The average length of proteins in each class: \n{g_avg_length_per_class.to_string(header=False)} \n")
print(f"Maximum and minimum length of proteins in each class: \n{g_min_max_per_class.to_string()} \n")

print("GRAM-NEGATIVE BACTERIA")
print(f"Number of proteins: {n_num_proteins} \n")
print(f"Number of Labels: {n_num_labels} \n")
print(f"Number of proteins in each class:\n{n_formatting_help}\n")
print(f"The average length of proteins in each class: \n{n_avg_length_per_class.to_string(header=False)} \n")
print(f"Maximum and minimum length of proteins in each class: \n{n_min_max_per_class.to_string()}")
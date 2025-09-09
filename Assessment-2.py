import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st


#Loading datasets
d1 = pd.read_csv("/Users/sagar/Desktop/git hit 140 /HIT140-Assignment2/dataset1.csv")
d2 = pd.read_csv("/Users/sagar/Desktop/git hit 140 /HIT140-Assignment2/dataset2.csv")

#Peek on datasets
print(d1.head())
print(d1.info())

#Finding unique values of "habit" column.
print(d1["habit"].unique())
"""
    Number of unique values of "habit" column is huge so we cannot decide using this column.
"""

print("#"*100)
print(d1["habit"].value_counts())
print("#"*100)
print(d1.describe())
print("#"*100)
print(d1.isnull().sum())
"""
    Dataset 1 has 41 null values in in habit column, so it needs to be filled with mode value of the column.    
"""
d1["habit"].fillna(d1["habit"].mode()[0],inplace=True)
print("#"*100)
print(d1.isnull().sum())
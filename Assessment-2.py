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

#Select only numeric data type  
numeric_df = d1.select_dtypes(include=['int64','float64'])
print(numeric_df)

#Plot correlation of numeric data types columns
plt.figure(figsize = (8,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation among numeric data types")
plt.show()

"""
    Correlation matrix shows that 'bat_landing_to_food' and 'risk' has positive correlation which we can discuss about their relationship about bat's behaviour.
    Similarly, risk and reward has negative correlation that means most of the risk taking rats were not successful.
    

"""

#Relatioinship between 'risk' and 'reward' columns
ct = pd.crosstab(d1['risk'],d1['reward'])
print(ct)

ct.plot(kind='bar', stacked=False)
plt.xlabel('Risk (0=avoid, 1=take)')
plt.ylabel('Count')
plt.title('Counts of Reward outcomes by Risk group')
plt.legend(title='Reward')
plt.show()

"""
    From the figure it is clear that when risk = 0, reward is high and when risk = 1, reward is high.
    That means bats take risk, they don't get food.
"""


# Analysis of feature 'bat_landing_to_food'
print("Average time bats approach food after landing in seconds", d1['bat_landing_to_food'].describe())

#Check for outliers in 'bat_landing_to_food' using boxplot

plt.figure(figsize=(8,8))
plt.boxplot(d1['bat_landing_to_food'])
plt.title("Outliers in 'bat_landing_to_food'")
plt.ylabel("seconds")
plt.show()

"""
    There are huge number of outliers in bat_landing_to_food let's remove them to clean the data. These outliers may affect our final result.
"""


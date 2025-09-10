import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st


#Loading datasets
d1 = pd.read_csv("dataset1.csv")
d2 = pd.read_csv("dataset2.csv")

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
    There are huge number of outliers in bat_landing_to_food let's remove them to clean the data. These outliers may affect our final result.
"""
#Cleaning outliers
Q1 = d1['bat_landing_to_food'].quantile(0.25)
Q3 = d1['bat_landing_to_food'].quantile(0.75)
IQR =  Q3 - Q1

lower_bound = Q1 - 1.5*IQR
upper_bound = Q1 + 1.5 * IQR

filtered_d1 = d1[(d1['bat_landing_to_food'] >= lower_bound) & (d1['bat_landing_to_food'] <= upper_bound)]
print(filtered_d1.describe())

"""
    Let's see if bats show hesitation to approach food and take risks or not. If they show more hesitation then they perceive bats as predators.
    Else they perceive rats as only competitors.
    For this we have to use t-test. TO compare if bat hesitates more or less.
   
    Since 'bat_landing_to_food' and 'risk' has higher correlation, let's calculate two sample t-test.
"""

#Case 1: bat_landing_to_food vs risk taken or not
sample1 = filtered_d1.loc[d1["risk"]==1, "bat_landing_to_food"]
sample2 = filtered_d1.loc[d1["risk"]==0, "bat_landing_to_food"]

mean1 = sample1.mean()
mean2 = sample2.mean()
std1 = sample1.std()
std2 = sample2.std()
n1 = len(sample1)
n2 = len(sample2)


"""
    Calculating two sample mean t-test, where sample1 has hesitation time of bats and taking risk.
    sample2 has hesitationi time of bats and not taking risk.
   
    Null Hypothesis: The number of risk taking bats are equal with number of bats not taking risk
    Alternative Hypothesis: The number of risk taking bats are not equal with number of risk not taking bats.
"""
t_stat, p_value = st.ttest_ind_from_stats(mean1, std1, n1, mean2, std2, n2, equal_var=False)
print("T-stats dataset 1",t_stat) #6.682
print("P-value dataset 1", p_value) #5.72x10^-11

print("Mean 1", mean1) #5.1686 seconds
print("Mean 2", mean2) #3.1159 seconds

"""
    Here p-value is too small which indicates that we reject the null hypothesis. Thus there are not
    equal number of bats that take risk and not taking risk. Analyzing mean we found that risk taking
    bats hesitates more before approaching food (5.1686 seconds) than risk not taking bats (3.11 secs)
   
    This hesitation means that bats hesitates more before approaching food platform when rats are
    present.
"""


#FOR DATASET 2

print(d2.info())

#Check correlation among numerical features.
plt.figure(figsize=(8,6))
numerical_columns = d2.select_dtypes(include=['int64','float64'])
sns.heatmap(numerical_columns.corr(), annot=True, cmap='coolwarm')
plt.title("Dataset 2 correlation among numerical features")
plt.show()

"""
    Correlation matrix shows that 'bat_landing_number' and 'food_availability' has high correlation.
    However, 'bat_landing_number' and 'rat_minutes' are negatively correlated.
"""

print(d2['food_availability'].describe())
ct = pd.crosstab(d2['food_availability'],d2['bat_landing_number'])

#Distribution of food availability and bat landing number
plt.figure(figsize=(8,6))
plt.scatter(d2['food_availability'],d2['bat_landing_number'])
plt.title("Distribution of 'food_availability' and 'bat_landing_number'")
plt.xlabel("Food Availability")
plt.ylabel("Bat_landing_number")
plt.show()


#Distribution of food availability and rat arrival number
plt.figure(figsize=(8,6))
plt.scatter(d2['rat_arrival_number'],d2['food_availability'])
plt.title("Distribution of 'food_availability' and 'rat_arrival_number'")
plt.xlabel("Rat Arrival Number")
plt.ylabel("Food Availability")
plt.show()

#Distribution of rat arrival number and bat landing number
plt.figure(figsize=(8,6))
plt.scatter(d2['bat_landing_number'], d2['rat_arrival_number'])
plt.title("Distribution of 'rat_arrival_number' and 'bat_landing_number'")
plt.xlabel("Bat Landing Number")
plt.ylabel("Rat Arrival Number")
plt.show()

#Case 2: Bat Landing number and Rat Minutes
"""
    Here we check how often bat lands when rats are present. 
    Using features 'bat_landing_number' and 'rat_minutes'
"""
sample1d2 = d2.loc[d2['rat_minutes'] > 0, 'bat_landing_number']
sample2d2 = d2.loc[d2['rat_minutes'] == 0, 'bat_landing_number']

#calculating mean and standard deviation of dataset2
mean1d2 = sample1d2.mean()
mean2d2 = sample2d2.mean()
std1d2 = sample1d2.std()
std2d2 = sample2d2.std()

n1d2 = len(sample1d2)
n2d2 = len(sample2d2)

"""
    Let's use two sample t-test to test if these two samples have equal mean or not
    Null Hypothesis:- Number of bat landing in platform when rats are present is equal to number
    of bats landing when rats are not present
    Alternate Hypothesis:- Number of bat landing in platform when rats are present is not equal to 
    number of bats landing when rats are not present.
"""
t2_value, p2_value = st.ttest_ind_from_stats(mean1d2, std1d2, n1d2, mean2d2, std2d2, n2d2, equal_var=False)
print("T2_value ", t2_value)
print("P2_Value ", p2_value)

"""
    Here t-value is -5.08. The -ve sign indicates that mean number of bat landing when rats
    are not present is greater than the mean number of bat landing when rats are present.
    And 'p_value' is 4.38x10^-7, this means the two mean are not equal (Reject Nulll Hypothesis). 
"""

#So, it came to the conclusion, that from dataset 1 bats take much time to land when taking risk.
#Also, from dataset 2 using t-test it is proved that when rats are present, less bat land on platform.
#So, this makes sense that bats are avoiding rats and perceive them as predators.


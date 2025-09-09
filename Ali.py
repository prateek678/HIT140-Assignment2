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
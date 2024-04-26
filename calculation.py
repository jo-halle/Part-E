import pandas as pd
from scipy import stats

# Assuming training_data is your DataFrame
import seaborn as sns
import matplotlib.pyplot as plt

# Define column names based on the dataset description
column_names = ['Suit1', 'Rank1', 'Suit2', 'Rank2', 'Suit3', 'Rank3', 'Suit4', 'Rank4', 'Suit5', 'Rank5', 'Class']

# Load the datasets
training_data = pd.read_csv('data/poker-hand-training-true.data', names=column_names)
testing_data = pd.read_csv('data/poker-hand-testing.data', names=column_names)

# Calculate basic statistics for 'Rank1', 'Rank2', 'Rank3'
stats_df = training_data[['Rank1', 'Rank2', 'Rank3', 'Rank4', 'Rank5']].describe()

# Calculate mode for 'Rank1', 'Rank2', 'Rank3'
mode_df = training_data[['Rank1', 'Rank2', 'Rank3', 'Rank4', 'Rank5']].mode().iloc[0]

# Add mode to the statistics DataFrame
stats_df.loc['mode'] = mode_df

# Convert to integer for appropriate comparison
stats_df = stats_df.astype(int)

# Print the resulting statistics
print(stats_df)
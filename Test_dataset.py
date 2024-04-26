import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define column names based on the dataset description
column_names = ['Suit1', 'Rank1', 'Suit2', 'Rank2', 'Suit3', 'Rank3', 'Suit4', 'Rank4', 'Suit5', 'Rank5', 'Class']

# Load the datasets
training_data = pd.read_csv('data/poker-hand-training-true.data', names=column_names)
testing_data = pd.read_csv('data/poker-hand-testing.data', names=column_names)

# Display the first few rows to confirm proper loading
# print(training_data.head())
# print(testing_data.head())

# Convert suit and rank columns to 'category' dtype
categorical_columns = ['Suit1', 'Rank1', 'Suit2', 'Rank2', 'Suit3', 'Rank3', 'Suit4', 'Rank4', 'Suit5', 'Rank5']
for column in categorical_columns:
    training_data[column] = training_data[column].astype('category')
    testing_data[column] = testing_data[column].astype('category')

# Generate summary statistics for the training data
# print(training_data.describe())

# Check for and drop duplicates
# training_data = training_data.drop_duplicates()
# testing_data = testing_data.drop_duplicates()

# # Check for missing values
# print(training_data.isnull().sum())
# print(testing_data.isnull().sum())

# Correlation matrix
correlation_matrix = training_data.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()



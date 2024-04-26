import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define column names based on the dataset description
column_names = ['Suit1', 'Rank1', 'Suit2', 'Rank2', 'Suit3', 'Rank3', 'Suit4', 'Rank4', 'Suit5', 'Rank5', 'Class']

# Load the datasets
training_data = pd.read_csv('data/poker-hand-training-true.data', names=column_names)
testing_data = pd.read_csv('data/poker-hand-testing.data', names=column_names)

# Convert suit and rank columns to 'category' dtype
categorical_columns = ['Suit1', 'Rank1', 'Suit2', 'Rank2', 'Suit3', 'Rank3', 'Suit4', 'Rank4', 'Suit5', 'Rank5']
for column in categorical_columns:
    training_data[column] = training_data[column].astype('category')
    testing_data[column] = testing_data[column].astype('category')

# Correlation matrix
correlation_matrix = training_data.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

# # Calculate and print the distribution of data objects for each class in the training data
class_distribution = training_data['Class'].value_counts()
print("Class Distribution in Training Data:")
print(class_distribution)

# If you also want to see the distribution in the testing data, uncomment the following lines:
print("\nClass Distribution in Testing Data:")
print(testing_data['Class'].value_counts())

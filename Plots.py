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
    
# plt.figure(figsize=(10, 6))
# sns.countplot(x='Class', data=training_data, palette='viridis')  # Using 'viridis' palette
# plt.title('Distribution of Poker Hand Classes')
# plt.xlabel('Class')
# plt.ylabel('Frequency')
# plt.savefig('class_distribution.png')  # Save the figure
# plt.close()  # Close the plot to free up memory

# Scatter Plot of Rank1 vs. Rank2 Colored by Class
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Rank1', y='Rank3', hue='Class', data=training_data, palette='Paired')
plt.title('Scatter Plot of Rank1 vs. Rank3 Colored by Class')
plt.xlabel('Rank1')
plt.ylabel('Rank3')
plt.savefig('scatter_plot_rank1_Rank3.png')  # Save the figure
plt.close()  # Close the plot to free up memory

# # Selecting a subset for clarity in visualization and correcting pairplot
# sample_data = training_data[['Rank1', 'Rank2', 'Rank3', 'Class']].sample(frac=0.1)  # Adjust frac as needed

# # Correcting pairplot to handle categorical data correctly
# sns.pairplot(sample_data, hue='Class', vars=['Rank1', 'Rank2', 'Rank3'], diag_kind='hist', plot_kws={'alpha': 0.5}, palette='Paired')
# plt.suptitle('Pair Plot of Ranks Colored by Class', verticalalignment='top')
# plt.savefig('pair_plot_ranks_custom_colors.png')  # Save the figure
# plt.close()  # Close the plot to free up memory

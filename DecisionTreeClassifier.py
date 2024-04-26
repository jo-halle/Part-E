import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load your dataset
data = pd.read_csv('data/poker-hand-training-true.data', header=None)

# Assign features and target
X = data.iloc[:, :-1]  # all columns except the last one as features
y = data.iloc[:, -1]   # the last column as the target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to run an experiment with specified hyperparameters
def run_experiment(max_depth, min_samples_split=2, min_samples_leaf=1):
    # Initialize the Decision Tree Classifier with specified parameters
    dtree = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf, class_weight='balanced', random_state=42)

    # Train the model on the training data
    dtree.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = dtree.predict(X_test)

    # Print model evaluation
    print(f"Experiment with max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}")
    print("Decision Tree Classifier Report:\n", classification_report(y_test, predictions, zero_division=0))
    print("Accuracy:", accuracy_score(y_test, predictions))

# Running experiments with different hyperparameters
run_experiment(5)
run_experiment(10, 4)
run_experiment(15, 10, 4)

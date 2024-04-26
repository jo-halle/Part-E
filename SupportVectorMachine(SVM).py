import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('data/poker-hand-training-true.data', header=None)
X = data.iloc[:, :-1]  # all columns except the last one as features
y = data.iloc[:, -1]   # the last column as the target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# # SVM with RBF kernel, adjusted parameters, and class weight balanced
# svm_rbf = SVC(kernel='rbf', C=10, gamma=0.01, class_weight='balanced', random_state=42)
# svm_rbf.fit(X_train_scaled, y_train)
# predictions_rbf = svm_rbf.predict(X_test_scaled)
# # Evaluation
# print("Adjusted SVM RBF Classifier Report:\n", classification_report(y_test, predictions_rbf, zero_division=0))
# print("Accuracy:", accuracy_score(y_test, predictions_rbf))

# # Tweak Set 1: More Flexible Boundary
# svm_rbf_flexible = SVC(kernel='rbf', C=50, gamma=0.1, class_weight='balanced', random_state=42)
# svm_rbf_flexible.fit(X_train_scaled, y_train)
# predictions_rbf_flexible = svm_rbf_flexible.predict(X_test_scaled)
# print("SVM RBF Flexible Classifier Report:\n", classification_report(y_test, predictions_rbf_flexible, zero_division=0))
# print("Accuracy:", accuracy_score(y_test, predictions_rbf_flexible))

# Tweak Set 2: More Generalized Approach
svm_rbf_generalized = SVC(kernel='rbf', C=1, gamma=0.01, class_weight='balanced', random_state=42)
svm_rbf_generalized.fit(X_train_scaled, y_train)
predictions_rbf_generalized = svm_rbf_generalized.predict(X_test_scaled)
print("SVM RBF Generalized Classifier Report:\n", classification_report(y_test, predictions_rbf_generalized, zero_division=0))
print("Accuracy:", accuracy_score(y_test, predictions_rbf_generalized))


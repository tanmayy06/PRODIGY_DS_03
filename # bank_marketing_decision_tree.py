# bank_marketing_decision_tree.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

# Get script directory to save outputs locally
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load dataset
csv_path = os.path.join(script_dir, r'T:\task 3\bank.csv')  # Replace with your actual file name
df = pd.read_csv(csv_path,sep=';')

# Display basic info
print("Dataset shape:", df.shape)
print("First 5 rows:")
print(df.head())
print("\nMissing values:\n", df.isnull().sum())

# Encode categorical features
df_encoded = pd.get_dummies(df, drop_first=True)

# Split into features and target
X = df_encoded.drop('y_yes', axis=1) if 'y_yes' in df_encoded.columns else df_encoded.drop('y', axis=1)
y = df_encoded['y_yes'] if 'y_yes' in df_encoded.columns else df['y'].apply(lambda x: 1 if x == 'yes' else 0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot and save the tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, fontsize=8)
tree_path = os.path.join(script_dir, 'decision_tree_plot.png')
plt.savefig(tree_path)
print(f"\nDecision Tree plot saved to: {tree_path}")

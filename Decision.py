import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("anxiety_depression_dataset_200_clean.csv")

# Choose 4 numeric features (example: 2 PHQ + 2 GAD items)
X = df[['phq1', 'phq2', 'gad1', 'gad2']]

# Target variable (example: depression severity)
y = df['phq_severity']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict & evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Decision Tree Accuracy (using 4 features):", accuracy)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load your dataset (replace with your CSV file)
df = pd.read_csv("anxiety_depression_dataset_200_clean.csv")

# Example: use PHQ-9 and GAD-7 totals as features
X = df[["phq_total", "gad_total"]]

# Example: predict PHQ severity
y = df["phq_severity"]

# Encode labels (convert text labels to numbers)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Test KNN with 4 different values of k
k_values = [1, 3, 5, 7]
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"K={k} -> Accuracy: {acc:.3f}")

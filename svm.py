import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load your cleaned dataset
df = pd.read_csv("anxiety_depression_dataset_200_clean.csv")

# Features = PHQ + GAD items
phq_items = [f'phq{i}' for i in range(1,10)]
gad_items = [f'gad{i}' for i in range(1,8)]
X = df[phq_items + gad_items]

# Target = PHQ severity (example, you can swap to gad_severity)
y = df['phq_severity']

# Encode severity labels numerically
y = y.astype('category').cat.codes  # maps severity strings -> integers

# Test on 4 dataset sizes
sizes = [50, 100, 150, 200]

for n in sizes:
    X_sub = X.iloc[:n]
    y_sub = y.iloc[:n]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_sub, test_size=0.2, random_state=42
    )

    # Train SVM
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"SVM accuracy with {n} samples: {acc:.2f}")

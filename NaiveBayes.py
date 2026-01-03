import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("anxiety_depression_dataset_200_clean.csv")

# Features: PHQ and GAD items
features = [f'phq{i}' for i in range(1,10)] + [f'gad{i}' for i in range(1,8)]
X = df[features]

# Target: example with PHQ severity (could switch to gad_severity if you prefer)
y = df['phq_severity']

# Store results for 4 different splits
accuracies = []

for random_seed in [1, 11, 21, 31]:  # 4 different splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_seed
    )
    
    # Train Naive Bayes
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append((random_seed, acc))

# Print results
for seed, acc in accuracies:
    print(f"Random Seed {seed}: Accuracy = {acc:.3f}")

print(f"Average Accuracy: {sum(acc for _, acc in accuracies)/len(accuracies):.3f}")

import argparse
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from data_loader import load_cora_content

# Add file path from CLI
parser = argparse.ArgumentParser(description="Logistic Regression on Cora Dataset")
parser.add_argument('--content', type=str, required=True, help="Path to cora.content file")
args = parser.parse_args()

# Load data
X, y, paper_ids, label_to_index, index_to_label, df = load_cora_content(args.content)

# Initialize cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
all_preds = np.zeros_like(y)

# 10-fold cross-validation
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    all_preds[test_index] = y_pred  # Store predictions in original order

# Evaluate accuracy
accuracy = accuracy_score(y, all_preds)
print(f"Baseline Accuracy: {accuracy:.4f}")

# Convert predicted labels to strings
predicted_labels = [index_to_label[i] for i in all_preds]

# Save to TSV
with open("baseline_predictions.tsv", "w") as f:
    for paper_id, label in zip(paper_ids, predicted_labels):
        f.write(f"{paper_id}\t{label}\n")
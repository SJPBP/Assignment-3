# Step 1: Import necessary tools
# from sklearn.datasets import load_iris  # toy dataset to practice
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
)  # to split data

from sklearn.ensemble import RandomForestClassifier  # the model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)  # to evaluate

import pandas as pd
import numpy as np
from time import time
from Bio.SeqUtils.ProtParam import ProteinAnalysis


# load csv, use map to change numbered classes to named classes
g_data = pd.read_csv("g_data.csv", header=None)
y = g_data.iloc[:, 0].map(
    {
        1: "inner membrane proteins",
        2: "outer membrane proteins",
        3: "cytoplasm proteins",
        4: "extracell proteins",
    }
)
sequences = g_data.iloc[:, 3]

# Get full amino acid counts in a fixed order
amino_acids = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]


# extract features using biopython
def extract_features(seq: str) -> list:
    # Replace amino acid "U" to "C"
    X = ProteinAnalysis(str(seq).upper().replace("U", "C"))

    aa_occurrence = [
        X.count_amino_acids().get(aa, 0) for aa in amino_acids
    ]  # raw counts
    aa_composition = [
        X.amino_acids_percent.get(aa, 0.0) for aa in amino_acids
    ]  # percentage

    flexibility_avg = sum(X.flexibility()) / len(X.flexibility())

    return (
        aa_occurrence
        + aa_composition
        + [
            X.molecular_weight(),
            X.aromaticity(),
            X.instability_index(),
            X.isoelectric_point(),
            X.secondary_structure_fraction()[0],
            X.secondary_structure_fraction()[1],
            X.secondary_structure_fraction()[2],
            X.gravy(),
            flexibility_avg,
        ]
    )


feature_matrix = [extract_features(s) for s in sequences]
columns = (
    [f"{aa}_count" for aa in amino_acids]
    + [f"{aa}_percent" for aa in amino_acids]
    + [
        "molecular_weight",
        "aromaticity",
        "instability_index",
        "isoelectric_point",
        "helix_fraction",
        "sheet_fraction",
        "coil_fraction",
        "gravy",
        "flexibility_avg",
    ]
)
X = pd.DataFrame(feature_matrix, columns=columns)  # x is a df made from function

# printing for checking class distribution
vc = y.value_counts()
vc.index.name = None  # drop extra “0” index name
print("Class Balances:")
print(vc)

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 4: Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=120,
    random_state=42,
    max_depth=None,
    min_samples_split=2,
    class_weight="balanced",
)

start = time()  # Start timer
# Step 5: Train the model (fit it to the training data)
model.fit(X_train, y_train)
print(f"\nTrained in {time() - start:.2f}s")

# indices were not aligned somehow before so I'll do that here
y_test = y_test.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_pred = model.predict(X_test)

# save results to see T and F predictions in a csv
results = pd.DataFrame(
    {"True_Label": y_test, "Predicted": y_pred, "Is_Correct": y_test == y_pred}
)

# display accuracy and save results
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}")
# results = pd.concat([results, X_test], axis=1)
# results.to_csv("svm_predictions.csv", index=False)
# print("\nSaved fixed output to svm_predictions.csv")

start = time()
scores = cross_val_score(model, X, y, cv=7)  # 5-fold CV
print(f"Cross-Validation Accuracy: {scores.mean() * 100:.2f}")

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# assume you already have y_test, y_pred
cm = confusion_matrix(y_test, y_pred)
classes = model.classes_  # your four classes in order

# Loop over each class i:
for i, cls in enumerate(classes):
    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FP = cm[:, i].sum() - TP
    TN = cm.sum() - (TP + FN + FP)

    sensitivity = TP / (TP + FN) if (TP + FN) else np.nan
    specificity = TN / (TN + FP) if (TN + FP) else np.nan

    print(f"Class = {cls}")
    print(f"  Sensitivity (TPR) = {sensitivity:.3f}")
    print(f"  Specificity (TNR) = {specificity:.3f}\n")

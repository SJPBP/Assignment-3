from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from time import time

# csv to df and then update numbered classes to proper names
g_data = pd.read_csv("g_data.csv", header=None)
y = g_data.iloc[:, 0].map({
    1: "inner membrane proteins",
    2: "outer membrane proteins",
    3: "cytoplasm proteins",
    4: "extracell proteins"
})
sequences = g_data.iloc[:, 3]

amino_acids = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H",
    "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"
]

# extracting with biopython
def extract_features(seq):
    X = ProteinAnalysis(str(seq).upper().replace('U', 'C'))

    aa_counts = X.count_amino_acids()
    aa_percent = X.amino_acids_percent
    flexibility_avg = sum(X.flexibility()) / len(X.flexibility())

    features = (
        [aa_counts.get(aa, 0) for aa in amino_acids] +    # counts
        [aa_percent.get(aa, 0.0) for aa in amino_acids] +  # percentages
        [
            X.molecular_weight(),
            X.aromaticity(),
            X.instability_index(),
            X.isoelectric_point(),
            X.secondary_structure_fraction()[0],  # helix
            X.secondary_structure_fraction()[1],  # sheet
            X.secondary_structure_fraction()[2],  # coil
            X.gravy(), # with some mashed potatoes omg
            flexibility_avg
        ]
    )
    return features

# build feature matrix
feature_matrix = [extract_features(s) for s in sequences]
columns = (
    [f"{aa}_count" for aa in amino_acids] +
    [f"{aa}_percent" for aa in amino_acids] +
    [
        "molecular_weight", "aromaticity", "instability_index",
        "isoelectric_point", "helix_fraction", "sheet_fraction",
        "coil_fraction", "gravy", "flexibility_avg"
    ]
)
X = pd.DataFrame(feature_matrix, columns=columns)

# Print class distribution CAN BE DELETED LATER
vc = y.value_counts()
vc.index.name = None
print("Class Balances:")
print(vc)

# split data into train and test sets
# TO DO: consider k-fold cross validation since dataset is ~500 proteins
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# calculate class weights
class_weights = {label: (len(y_train) / (len(vc) * count)) for label, count in vc.items()}

# train LinearSVC
clf = LinearSVC(dual=False, max_iter=10000, random_state=42, class_weight=class_weights)
start = time()
clf.fit(X_train, y_train)
print(f"\nTrained in {time() - start:.2f}s")

# test the model
y_test = y_test.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_pred = clf.predict(X_test)

# save results in a csv for output checking CAN BE DELETED LATER
results = pd.DataFrame({
    'True_Label': y_test,
    'Predicted': y_pred,
    'Is_Correct': y_test == y_pred
})
results = pd.concat([results, X_test], axis=1)

print("Accuracy:", accuracy_score(y_test, y_pred))
results.to_csv('svm_predictions.csv', index=False)
print("\nSaved output to svm_predictions.csv")

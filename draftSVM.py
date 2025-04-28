from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from time import time

# load csv, use map to change numbered classes to named classes
g_data = pd.read_csv("g_data.csv", header=None)
y = g_data.iloc[:, 0].map({
    1: "inner membrane proteins",
    2: "outer membrane proteins",
    3: "cytoplasm proteins",
    4: "extracell proteins"
})
sequences = g_data.iloc[:, 3]

# extract features using biopython
def extract_features(seq):
    X = ProteinAnalysis(str(seq).upper().replace('U','C'))
    return [
        X.count_amino_acids().get('A',0),
        X.count_amino_acids().get('E',0),
        X.amino_acids_percent.get('K',0.0),
        X.amino_acids_percent.get('L',0.0),
        X.molecular_weight(),
        X.aromaticity(),
        X.instability_index(),
        X.isoelectric_point(),
        X.secondary_structure_fraction()[0]
    ]

feature_matrix = [extract_features(s) for s in sequences]
columns = ['A_count','E_count','K_percent','L_percent',
           'molecular_weight','aromaticity','instability_index',
           'isoelectric_point','helix_fraction']
X = pd.DataFrame(feature_matrix, columns=columns) # x is a df made from function

# printing for checking class distribution
vc = y.value_counts()
vc.index.name = None        # drop extra “0” index name
print("Class Balances:")
print(vc)

# split data 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# calculate class weights based on class distribution 
# (class weight for outer membrane proteins is low)
class_weights = {label: (len(y_train) / (len(vc) * count)) for label, count in vc.items()}

# make classifier and train with class weights
clf = LinearSVC(dual=False, max_iter=10000, random_state=42, class_weight=class_weights)
start = time()
clf.fit(X_train, y_train)
print(f"\nTrained in {time() - start:.2f}s")

# indices were not aligned somehow before so I'll do that here
y_test = y_test.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_pred  = clf.predict(X_test)

# save results to see T and F predictions in a csv
results = pd.DataFrame({
    'True_Label': y_test,
    'Predicted':  y_pred,
    'Is_Correct': y_test == y_pred
})

# display accuracy and save results
print("Accuracy:", accuracy_score(y_test, y_pred))
results = pd.concat([results, X_test], axis=1)
results.to_csv('svm_predictions.csv', index=False)
print("\nSaved fixed output to svm_predictions.csv")

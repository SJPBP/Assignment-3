from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef

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

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# calculate class weights
cw = y.value_counts()
cw.index.name = None
class_weights = {label: (len(y_train) / (len(cw) * count)) for label, count in cw.items()}

# train LinearSVC 
# not using kernel=linear bc it takes took long for some reason
clf = LinearSVC(dual=False, max_iter=10000, random_state=42, class_weight=class_weights)
clf.fit(X_train, y_train)

# test the model
y_test = y_test.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_pred = clf.predict(X_test)

# grab accuracy
acc = clf.score(X_test, y_test)

# grab confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

# grab sensitivity per classs
recall = recall_score(y_test, y_pred, average=None, labels=clf.classes_)

# grab MCC
mcc = matthews_corrcoef(y_test, y_pred)

# grab pecificity per class = TN / (TN + FP)
specificity = []
for i in range(len(cm)):
    tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
    fp = cm[:, i].sum() - cm[i, i]
    specificity.append(tn / (tn + fp) if (tn + fp) != 0 else 0.0)

# Output results
print("\nEvaluation Metrics")
print("==================")
print(f"Accuracy: {acc:.4f}")
print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}\n")

print("Class-wise Metrics:")
print("-------------------")
for idx, cls in enumerate(clf.classes_):
    print(f"Class: {cls}")
    print(f"  Sensitivity (Recall): {recall[idx]:.4f}")
    print(f"  Specificity:          {specificity[idx]:.4f}\n")

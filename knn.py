# Jessica Reyes - Machine Learning Project
# you need these packages: scikit-learn, biopython, matplotlib, pandas, numpy
# install using "pip install scikit-learn biopython matplotlib pandas numpy"

# import all packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # need for visualization graphs
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# import g_data using csv reader
df = pd.read_csv("g_data.csv")
sequence_col = df.columns[3]  # column with amino acid sequences


# function for extracting features from aa sequences
def extract_features(seq):
    seq = seq.replace('U', 'C')
    X = ProteinAnalysis(seq)

    aa_percent = X.amino_acids_percent
    dipeptide_count = {
        f"{aa1}{aa2}": seq.count(f"{aa1}{aa2}")
        for aa1 in 'ACDEFGHIKLMNPQRSTVWY'
        for aa2 in 'ACDEFGHIKLMNPQRSTVWY'
    }
    dipeptide_percent = {k: v / len(seq) for k, v in dipeptide_count.items()}
    pseAAC = {f"PseAAC_{aa}": aa_percent.get(aa, 0) for aa in 'ACDEFGHIKLMNPQRSTVWY'}

    features = {
        **aa_percent,
        **dipeptide_percent,
        **pseAAC,
        'molecular_weight': X.molecular_weight(),
        'aromaticity': X.aromaticity(),
        'instability_index': X.instability_index(),
        'isoelectric_point': X.isoelectric_point(),
        'helix_fraction': X.secondary_structure_fraction()[0]
    }
    return pd.Series(features)

# extract features from data and use it to build feature matrix
df_features = df[sequence_col].apply(extract_features)
X = df_features.fillna(0) # fill in any blanks
y = pd.Categorical(df[df.columns[0]]).codes # specify first column in csv as categories

# train and test split, test size 0.2 = 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# create pipeline and parameter grid to improve accuracy
# apparently they call things that automatically apply these scaling features to classifiers a "pipeline"
pipeline = Pipeline([
    ('scaler', StandardScaler()), # this scaler ensures all features contribute equally to KNN algorithm 
    ('knn', KNeighborsClassifier())
])

# parameter grid used for hyperparameter tuning, improves accuracy
# tells model to try values between 1-20 for n_neighbors
# based on rule of thumb that k should equal up to the square root of the total number of samples
param_grid = {
    'knn__n_neighbors': list(range(1, 21))
} # need for GridSearchCV

# grid search with stratified k-fold cross-validation for improved accuracy
# mainly used with 5 folds like below
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# print best parameters and cross-validation score
print("\nBest k (n_neighbors):", grid_search.best_params_['knn__n_neighbors'])
print(f"Best Cross-Validated Accuracy: {grid_search.best_score_ * 100:.2f}%")

# predict data based on best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# accuracy score based on number of correctly classified samples 
# over the total number of samples * 100 = percentage
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# classification report
class_labels = [1, 2, 3, 4] # wanted classes to be named 1-4 instead of 0-3
print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=[0,1,2,3], target_names=[f"Class {i}" for i in class_labels]))

# need confusion matrix for other accuracy metrics like sensitivity
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# sensitivity - measures proportion of correctly identified positives
# specificity - measures proportion of correctly identified negatives
TP = cm.diagonal() # true positive
FN = cm.sum(axis=1) - TP # false negative
FP = cm.sum(axis=0) - TP # false positive
TN = cm.sum() - (TP + FP + FN) # true negative

sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

# lists sensitivity and specificity for each class
for i, (sens, spec) in enumerate(zip(sensitivity, specificity)):
    print(f"\nClass {i+1}:")
    print(f"  Sensitivity: {sens:.2f}")
    print(f"  Specificity: {spec:.2f}")

# scatter plot of actual versus predicted results
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual', alpha=0.6)
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted', alpha=0.6)
plt.title('Actual vs. Predicted Classes')
plt.xlabel('Sample Index')
plt.ylabel('Class Label')
plt.yticks(np.unique(y), [f"Class {i+1}" for i in np.unique(y)])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
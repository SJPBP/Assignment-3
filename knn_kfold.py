# Jessica Reyes - Machine Learning Project
# Must download sklearn and biopython for this to work

import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# load g_data
df = pd.read_csv("g_data.csv")
sequence_col = df.columns[3]

def extract_features(seq):
    # c is chemically similar to u. raman said it was ok to either replace or drop!!
    seq = seq.replace('U', 'C')  

    X = ProteinAnalysis(seq)

    # calculates percentage of all amino acids at once instead of listing them out
    aa_percent = X.amino_acids_percent 
    
    # counts all dipeptides in the amino acid sequences
    dipeptide_count = { 
        f"{aa1}{aa2}": seq.count(f"{aa1}{aa2}")
        for aa1 in 'ACDEFGHIKLMNPQRSTVWY'
        for aa2 in 'ACDEFGHIKLMNPQRSTVWY'
    } 

    # calculates percentage of all dipeptides
    dipeptide_percent = {k: v / len(seq) for k, v in dipeptide_count.items()} 

    # pseudo-amino acid composition
    pseAAC = {f"PseAAC_{aa}": aa_percent.get(aa, 0) for aa in 'ACDEFGHIKLMNPQRSTVWY'} 

    # all the features i'm using to train my KNN classifier model
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

# use feature extraction function
df_features = df[sequence_col].apply(extract_features)
X = df_features.fillna(0)  # advised to fill any potential NaNs
y = pd.Categorical(df[df.columns[0]]).codes  # changes to numerical value

# scaling for improved accuracy
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# hyper parameter tuning for improved accuracy
param_grid = {
    'knn__n_neighbors': list(range(1, 21))  # test k from 1 to 20
}

# k-fold cross validation using GridSearchCV with StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)

# grid search with standardscaler and KNN for fitting model
grid_search.fit(X, y)

print("\nBest k (n_neighbors):", grid_search.best_params_['knn__n_neighbors'])
print(f"Best Cross-Validated Accuracy: {grid_search.best_score_ * 100:.2f}%")
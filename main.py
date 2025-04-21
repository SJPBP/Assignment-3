import pandas as pd

# Step 1: Import necessary tools
from sklearn.datasets import load_iris  # toy dataset to practice
from sklearn.model_selection import train_test_split  # to split data
from sklearn.ensemble import RandomForestClassifier  # the model
from sklearn.metrics import accuracy_score  # to evaluate

# Get postive gram data
df = pd.read_csv("./g_data.csv")

# Step 2: Load the dataset
# For Features take Amino Acid and >code
# Drop Column 0, 1 and start from 2
X = df.iloc[:,2:]

# Get all the rows from Column 0
y = df.iloc[:,0]

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Create the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 5: Train the model (fit it to the training data)
model.fit(X_train, y_train)

# Step 6: Make predictions on the test data
y_pred = model.predict(X_test)


# Step 7: Measure how well it did
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("Final.csv")

print("Dataset Loaded Successfully")

# Drop non-numeric column
df = df.drop("name", axis=1)

# Convert Yes/No to 1/0
df = df.replace({"Yes": 1, "No": 0})

# Convert True/False to 1/0
df = df.replace({True: 1, False: 0})

# Target column
y = df["class"]

# Feature columns
X = df.drop("class", axis=1)

# Fill missing values
X = X.fillna(0)

# Convert everything to numeric (IMPORTANT)
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "malware_model.pkl")

print("Model trained and saved successfully!")
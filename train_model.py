import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Load dataset
df = pd.read_csv("data/churn.csv")

# Show first rows
print(df.head())

# Dataset info
print(df.info())

# Missing values
print(df.isnull().sum())

# Remove useless columns
df.drop(
    ["Year", "CustomerId", "Surname"],
    axis=1,
    inplace=True
)

# Encode categorical variables
df = pd.get_dummies(
    df,
    columns=["Geography", "Gender"],
    drop_first=True
)

print(df.columns)

# Features and target
X = df.drop("Exited", axis=1)

y = df["Exited"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, pred))

# Classification report
print(classification_report(y_test, pred))

# Confusion matrix
print(confusion_matrix(y_test, pred))

# Feature importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)

# Save model
joblib.dump(model, "models/churn_model.pkl")

joblib.dump(scaler, "models/scaler.pkl")

print("Model Saved Successfully")

import matplotlib.pyplot as plt

importance = importance.head(10)

plt.figure(figsize=(10,5))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.title("Top Feature Importance")

plt.xlabel("Importance Score")

plt.ylabel("Features")

plt.show()
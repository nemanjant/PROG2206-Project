import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load labeled data
data = pd.read_csv("data/labels_with_features.csv")

# Features and labels
X = data[["area", "perimeter", "eccentricity", "solidity", "extent", "convex_area", "aspect_ratio", "compactness", "contrast"]]
y = data["type"]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, class_weight="balanced")
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(clf, "models/classifier/leaf_stem_classifier.pth")
print("Trained model saved to models/classifier/leaf_stem_classifier.pth")
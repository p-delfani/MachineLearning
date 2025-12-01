# Install required libraries
# pip install pandas scikit-learn xgboost matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# ----------------------------
# Load NSL-KDD dataset
# ----------------------------
url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
columns = [f"feature_{i}" for i in range(41)] + ["label"]
data = pd.read_csv(url, names=columns)

# ----------------------------
# Binary label encoding
# normal = 0, attack = 1
# ----------------------------
data["label"] = data["label"].apply(lambda x: 0 if x=="normal" else 1)
y = data["label"].values

# ----------------------------
# Feature scaling
# ----------------------------
X = data.drop("label", axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# Split dataset
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Build XGBoost model
# ----------------------------
model = XGBClassifier(
    n_estimators=250,
    max_depth=8,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="logloss"
)

# Train model
model.fit(X_train, y_train)

# ----------------------------
# Predictions
# ----------------------------
y_pred = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)

# ----------------------------
# Evaluation
# ----------------------------
print("\nXGBoost IDS Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# Confusion Matrix
# ----------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - XGBoost IDS")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ----------------------------
# Feature Importance Plot
# ----------------------------
plt.figure(figsize=(10,6))
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.title("XGBoost Feature Importance")
plt.xlabel("Feature Index")
plt.ylabel("Importance Score")
plt.show()

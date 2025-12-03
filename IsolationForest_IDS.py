# Install required libraries
# pip install pandas scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, accuracy_score

# ----------------------------
# Load NSL-KDD Dataset
# ----------------------------
url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
columns = [f"feature_{i}" for i in range(41)] + ["label"]
data = pd.read_csv(url, names=columns)

# ----------------------------
# Binary Label Encoding
# Normal = 0, Attack = 1
# ----------------------------
data["label"] = data["label"].apply(lambda x: 0 if x == "normal" else 1)
y_true = data["label"].values

# ----------------------------
# Feature Scaling
# ----------------------------
X = data.drop("label", axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# Train Isolation Forest 
# Only requires normal data to learn normal pattern
# ----------------------------
iso = IsolationForest(
    n_estimators=300,
    contamination=0.15,   # Estimated attack percentage
    random_state=42
)

iso.fit(X_scaled)

# Predict anomalies
y_pred = iso.predict(X_scaled)

# Isolation Forest output:
#  1  → normal
# -1  → anomaly
y_pred = np.where(y_pred == 1, 0, 1)

# ----------------------------
# Evaluation
# ----------------------------
accuracy = accuracy_score(y_true, y_pred)
print("Isolation Forest IDS Accuracy:", accuracy)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap="Reds", fmt="d")
plt.title("Confusion Matrix - Isolation Forest IDS")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ----------------------------
# Visualizing anomaly scores
# ----------------------------
scores = iso.decision_function(X_scaled)

plt.figure(figsize=(8,4))
plt.plot(scores, label="Anomaly Score")
plt.axhline(np.percentile(scores, 10), color="gray", linestyle="--", label="Threshold")
plt.title("Isolation Forest Anomaly Scores")
plt.ylabel("Score")
plt.xlabel("Sample Index")
plt.legend()
plt.show()

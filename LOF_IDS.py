# Install required libraries
# pip install pandas scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Load NSL-KDD dataset
# ----------------------------
url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
columns = [f"feature_{i}" for i in range(41)] + ["label"]
data = pd.read_csv(url, names=columns)

# ----------------------------
# Binary label encoding
# Normal = 0, Attack = 1
# ----------------------------
data["label"] = data["label"].apply(lambda x: 0 if x == "normal" else 1)
y_true = data["label"].values

# ----------------------------
# Feature scaling
# ----------------------------
X = data.drop("label", axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# Train Local Outlier Factor (LOF)
# ----------------------------
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.15,  # Estimated attack percentage
    novelty=False
)

# Fit_predict outputs -1 for anomaly, 1 for normal
y_pred = lof.fit_predict(X_scaled)
y_pred = np.where(y_pred == 1, 0, 1)  # convert to 0=normal, 1=attack

# ----------------------------
# Evaluation
# ----------------------------
accuracy = accuracy_score(y_true, y_pred)
print("LOF IDS Accuracy:", accuracy)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap="Purples", fmt="d")
plt.title("Confusion Matrix - LOF IDS")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ----------------------------
# Visualize LOF negative_outlier_factor_
# ----------------------------
scores = -lof.negative_outlier_factor_
plt.figure(figsize=(8,4))
plt.plot(scores, label="LOF Anomaly Score")
plt.axhline(np.percentile(scores, 85), color="red", linestyle="--", label="Threshold")
plt.title("LOF Anomaly Scores")
plt.ylabel("Score")
plt.xlabel("Sample Index")
plt.legend()
plt.show()

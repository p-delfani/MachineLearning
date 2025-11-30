# Install required libraries
# pip install pandas scikit-learn matplotlib seaborn

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
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
# Encode labels for evaluation
# ----------------------------
# Normal=0, Attack=1
data["label"] = data["label"].apply(lambda x: 0 if x=="normal" else 1)
y_true = data["label"].values

# ----------------------------
# Scale features
# ----------------------------
X = data.drop("label", axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# Train One-Class SVM on only normal data
# ----------------------------
X_normal = X_scaled[y_true==0]
oc_svm = OneClassSVM(nu=0.05, kernel="rbf", gamma=0.05)
oc_svm.fit(X_normal)

# ----------------------------
# Predict anomalies
# ----------------------------
y_pred = oc_svm.predict(X_scaled)
# One-Class SVM outputs: 1 for normal, -1 for anomaly
y_pred = np.where(y_pred==1, 0, 1)

# ----------------------------
# Evaluation
# ----------------------------
print("One-Class SVM Accuracy:", accuracy_score(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges")
plt.title("Confusion Matrix - One-Class SVM IDS")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

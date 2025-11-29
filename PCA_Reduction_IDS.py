# Install required libraries
# pip install pandas scikit-learn matplotlib seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# --------------------------
# Load dataset
# --------------------------
url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
columns = [f"feature_{i}" for i in range(41)] + ["label"]
data = pd.read_csv(url, names=columns)

# --------------------------
# Encode labels
# --------------------------
data["label"] = data["label"].apply(lambda x: "normal" if x == "normal" else "attack")
encoder = LabelEncoder()
y = encoder.fit_transform(data["label"])

# --------------------------
# Feature scaling
# --------------------------
X = data.drop("label", axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------
# PCA dimensionality reduction
# --------------------------
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# --------------------------
# Train-test split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

# --------------------------
# Train Random Forest
# --------------------------
rf = RandomForestClassifier(n_estimators=120, random_state=42)
rf.fit(X_train, y_train)

# --------------------------
# Prediction and accuracy
# --------------------------
y_pred = rf.predict(X_test)
print(f"PCA + RF Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# --------------------------
# Confusion Matrix
# --------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap="coolwarm", fmt="d")
plt.title("Confusion Matrix - PCA + Random Forest IDS")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

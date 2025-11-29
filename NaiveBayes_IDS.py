# Install required libraries
# pip install pandas scikit-learn seaborn matplotlib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------
# Load dataset
# --------------------------
url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
columns = [f"feature_{i}" for i in range(41)] + ["label"]
data = pd.read_csv(url, names=columns)

# --------------------------
# Label encoding
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
# Train-test split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------
# Train Naive Bayes model
# --------------------------
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# --------------------------
# Predictions
# --------------------------
y_pred = nb_model.predict(X_test)

print(f"Naive Bayes Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# --------------------------
# Confusion Matrix
# --------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix - Naive Bayes IDS")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

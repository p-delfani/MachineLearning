# Install required libraries
# pip install pandas scikit-learn matplotlib seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --------------------------
# Load NSL-KDD dataset
# --------------------------
url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
columns = [f"feature_{i}" for i in range(41)] + ["label"]
data = pd.read_csv(url, names=columns)

# --------------------------
# Encode binary labels
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
# Create and train KNN model
# --------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# --------------------------
# Predictions and accuracy
# --------------------------
y_pred = knn.predict(X_test)
print(f"KNN Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# --------------------------
# Confusion matrix visualization
# --------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="OrRd",
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("Confusion Matrix - KNN IDS")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

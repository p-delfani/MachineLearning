# Install necessary libraries
# pip install pandas scikit-learn matplotlib seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --------------------------
# Load NSL-KDD dataset
# --------------------------
url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
columns = [f'feature_{i}' for i in range(41)] + ['label']
data = pd.read_csv(url, names=columns)

# --------------------------
# Encode labels (binary classification)
# --------------------------
# Convert all attacks into one class 'attack', keep normal as 'normal'
data['label'] = data['label'].apply(lambda x: 'normal' if x=='normal' else 'attack')
encoder = LabelEncoder()
y = encoder.fit_transform(data['label'])

# --------------------------
# Feature scaling
# --------------------------
X = data.drop('label', axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------
# Split data into train and test
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------
# Train Random Forest Classifier
# --------------------------
rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
rf_model.fit(X_train, y_train)

# --------------------------
# Predictions and Evaluation
# --------------------------
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Test Accuracy: {accuracy:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# --------------------------
# Confusion Matrix Visualization
# --------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Random Forest IDS')
plt.show()

# --------------------------
# Feature Importance Visualization
# --------------------------
importances = rf_model.feature_importances_
indices = importances.argsort()[::-1]  # sort descending

plt.figure(figsize=(12,6))
plt.bar(range(10), importances[indices[:10]], align='center')
plt.xticks(range(10), [f'feature_{i}' for i in indices[:10]], rotation=45)
plt.ylabel('Importance')
plt.title('Top 10 Important Features - Random Forest IDS')
plt.show()

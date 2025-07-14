import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib  # For saving/loading the model

# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 2. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Train multiple models and select the best
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
    "Support Vector Machine": SVC()
}

best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name} Accuracy: {score:.2f}")
    if score > best_score:
        best_score = score
        best_model = model

print(f"\nBest Model: {type(best_model).__name__} with Accuracy: {best_score:.2f}\n")

# 4. Classification report
y_pred = best_model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# 5. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 6. Visualize the Decision Tree if applicable
if isinstance(best_model, DecisionTreeClassifier):
    plt.figure(figsize=(10,6))
    plot_tree(best_model, feature_names=feature_names, class_names=target_names, filled=True)
    plt.title("Decision Tree Visualization")
    plt.show()

# 7. Predict on new samples
new_samples = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.7, 3.1, 4.7, 1.5],
    [7.2, 3.0, 5.8, 1.6]
])
predictions = best_model.predict(new_samples)
for i, pred in enumerate(predictions):
    print(f"Sample {i+1} prediction: {target_names[pred]}")

# 8. Save the best model
joblib.dump(best_model, "best_model.pkl")
print("\nModel saved as 'best_model.pkl'")

# 9. Load the saved model and predict
loaded_model = joblib.load("best_model.pkl")
print(f"Reloaded model prediction: {target_names[loaded_model.predict([new_samples[0]])[0]]}")

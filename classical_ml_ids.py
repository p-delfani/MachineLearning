"""
Classical Machine Learning Models for Intrusion Detection

Includes:
- Logistic Regression
- Random Forest
- Evaluation metrics
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_logistic_regression(X_train, y_train):
    """Train a logistic regression classifier."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Train a random forest classifier."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

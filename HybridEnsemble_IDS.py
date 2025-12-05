# Install required libraries
# pip install pandas scikit-learn xgboost tensorflow matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ----------------------------
# Load NSL-KDD dataset
# ----------------------------
url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
columns = [f"feature_{i}" for i in range(41)] + ["label"]
data = pd.read_csv(url, names=columns)

# ----------------------------
# Encode labels (Multi-Class)
# ----------------------------
# Categories: normal, dos, probe, r2l, u2r
dos = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'udpstorm', 'apache2', 'processtable', 'worm']
probe = ['satan', 'ipsweep', 'nmap', 'portsweep', 'mscan', 'saint']
r2l = ['guess_passwd','ftp_write','imap','phf','multihop','warezmaster','warezclient','spy','xlock','xsnoop','snmpguess','snmpgetattack','httptunnel','sendmail','named']
u2r = ['buffer_overflow','loadmodule','rootkit','perl','sqlattack','xterm','ps']

def categorize_attack(label):
    if label == 'normal':
        return 'normal'
    elif label in dos:
        return 'dos'
    elif label in probe:
        return 'probe'
    elif label in r2l:
        return 'r2l'
    elif label in u2r:
        return 'u2r'
    else:
        return 'other'

data['category'] = data['label'].apply(categorize_attack)
encoder = LabelEncoder()
y = encoder.fit_transform(data['category'])

# ----------------------------
# Feature scaling
# ----------------------------
X = data.drop(['label','category'], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# Train-Test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Model 1: Random Forest
# ----------------------------
rf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_acc:.4f}")

# ----------------------------
# Model 2: XGBoost
# ----------------------------
xgb = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
print(f"XGBoost Accuracy: {xgb_acc:.4f}")

# ----------------------------
# Model 3: Neural Network
# ----------------------------
nn = Sequential()
nn.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
nn.add(Dropout(0.3))
nn.add(Dense(32, activation='relu'))
nn.add(Dense(len(np.unique(y)), activation='softmax'))

nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
nn.fit(X_train, y_train, epochs=25, batch_size=64, validation_split=0.2, verbose=1)

nn_pred = np.argmax(nn.predict(X_test), axis=1)
nn_acc = accuracy_score(y_test, nn_pred)
print(f"Neural Network Accuracy: {nn_acc:.4f}")

# ----------------------------
# Ensemble Voting (Majority Vote)
# ----------------------------
ensemble_pred = []
for i in range(len(y_test)):
    votes = [rf_pred[i], xgb_pred[i], nn_pred[i]]
    ensemble_pred.append(max(set(votes), key=votes.count))

ensemble_acc = accuracy_score(y_test, ensemble_pred)
print(f"Hybrid Ensemble Accuracy: {ensemble_acc:.4f}")

# ----------------------------
# Confusion Matrix for Ensemble
# ----------------------------
cm = confusion_matrix(y_test, ensemble_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=encoder.classes_, yticklabels=encoder.classes_, cmap='Blues')
plt.title("Confusion Matrix - Hybrid Ensemble IDS")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ----------------------------
# Feature Importance from Random Forest
# ----------------------------
importances = rf.feature_importances_
indices = importances.argsort()[::-1]
plt.figure(figsize=(12,6))
plt.bar(range(10), importances[indices[:10]], align='center')
plt.xticks(range(10), [f'feature_{i}' for i in indices[:10]], rotation=45)
plt.ylabel('Importance')
plt.title('Top 10 Important Features - Random Forest in Ensemble')
plt.show()

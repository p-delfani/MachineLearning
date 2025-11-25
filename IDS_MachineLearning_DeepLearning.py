# Install necessary libraries
# pip install pandas scikit-learn tensorflow matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# --------------------------
# Load NSL-KDD dataset
# --------------------------
url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
columns = [f'feature_{i}' for i in range(41)] + ['label']
data = pd.read_csv(url, names=columns)

# --------------------------
# Encode labels (multi-class)
# --------------------------
# We will classify attacks into categories:
# 'normal', 'dos', 'probe', 'r2l', 'u2r'
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

# Encode categorical labels
encoder = LabelEncoder()
y = encoder.fit_transform(data['category'])

# Features
X = data.drop(['label', 'category'], axis=1)

# Scale features for neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------
# Split data into train and test
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------
# Build deep learning model
# --------------------------
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y)), activation='softmax')  # multi-class output
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# --------------------------
# Evaluate the model
# --------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Predict classes
y_pred = np.argmax(model.predict(X_test), axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# --------------------------
# Confusion Matrix
# --------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=encoder.classes_, yticklabels=encoder.classes_, cmap='Blues')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()

# --------------------------
# Plot training history
# --------------------------
plt.figure(figsize=(10,4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training History')
plt.legend()
plt.show()

# Install required libraries
# pip install pandas scikit-learn tensorflow matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Flatten, Reshape

# ----------------------------
# Load NSL-KDD dataset
# ----------------------------
url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
columns = [f"feature_{i}" for i in range(41)] + ["label"]
data = pd.read_csv(url, names=columns)

# ----------------------------
# Encode labels (binary)
# ----------------------------
data["label"] = data["label"].apply(lambda x: 0 if x=="normal" else 1)
y = data["label"].values

# ----------------------------
# Scale features
# ----------------------------
X = data.drop("label", axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for Conv1D input: (samples, timesteps, features)
X_cnn_lstm = X_scaled.reshape((X_scaled.shape[0], 41, 1))

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_cnn_lstm, y, test_size=0.2, random_state=42
)

# ----------------------------
# Build CNN + LSTM model
# ----------------------------
model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation="relu", input_shape=(41,1)))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, epochs=15, batch_size=64, validation_split=0.2)

# ----------------------------
# Evaluate model
# ----------------------------
y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\nCNN+LSTM IDS Classification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm")
plt.title("Confusion Matrix - CNN+LSTM IDS")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

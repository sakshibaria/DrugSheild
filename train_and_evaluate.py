import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

# ===============================
# REPRODUCIBILITY
# ===============================
np.random.seed(42)
tf.random.set_seed(42)

# ===============================
# LOAD DATA
# ===============================
X = np.load("X_features.npy")
y = np.load("y_labels.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

# ===============================
# STANDARDIZE FEATURES  🔥 IMPORTANT
# ===============================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ===============================
# TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ===============================
# CLASS WEIGHTS
# ===============================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = {
    0: class_weights[0],
    1: class_weights[1]
}

print("Class weights:", class_weight_dict)

# ===============================
# BUILD IMPROVED CLASSIFIER
# ===============================
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation="relu", input_shape=(X.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================
# CALLBACKS (SMART TRAINING)
# ===============================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    verbose=1
)

# ===============================
# TRAIN MODEL
# ===============================
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=60,
    batch_size=16,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ===============================
# EVALUATION
# ===============================
y_pred_prob = model.predict(X_test)

# 🔥 Try slightly adjusted threshold
threshold = 0.48
y_pred = (y_pred_prob > threshold).astype(int).ravel()

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Non-Drug", "Drug"]
    )
)

# ===============================
# SAVE TRAINED MODEL
# ===============================
os.makedirs("model", exist_ok=True)
model.save("model/image_classifier.h5")
print("\nMODEL SAVED SUCCESSFULLY → model/image_classifier.h5")

# ===============================
# CONFUSION MATRIX PLOT
# ===============================
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-Drug", "Drug"],
    yticklabels=["Non-Drug", "Drug"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

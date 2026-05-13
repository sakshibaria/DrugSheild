import os
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMAGE_SIZE = (224, 224)
DATASET_DIR = "labeled_dataset"

X = []
y = []

model = VGG16(weights="imagenet", include_top=False)
print("✅ VGG16 loaded")

label_map = {"drug": 1, "non_drug": 0}

for label_name in label_map:
    folder_path = os.path.join(DATASET_DIR, label_name)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img = load_img(img_path, target_size=IMAGE_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        features = model.predict(img_array, verbose=0)
        X.append(features.flatten())
        y.append(label_map[label_name])

X = np.array(X)
y = np.array(y)

np.save("X_features.npy", X)
np.save("y_labels.npy", y)

print("🎉 Feature extraction completed")
print("X shape:", X.shape)
print("y shape:", y.shape)

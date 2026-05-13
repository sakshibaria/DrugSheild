import os
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -------- CONFIG --------
IMAGE_FOLDER = "deduplicated_dataset"
IMAGE_SIZE = (224, 224)
FEATURES_OUTPUT = "vgg16_features.npy"
FILENAMES_OUTPUT = "image_names.npy"
# ------------------------

# Load VGG16 once
model = VGG16(weights="imagenet", include_top=False)
print("✅ VGG16 model loaded")

features_list = []
filenames_list = []

for filename in os.listdir(IMAGE_FOLDER):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(IMAGE_FOLDER, filename)

    try:
        img = load_img(image_path, target_size=IMAGE_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        features = model.predict(img_array, verbose=0)

        # Flatten (1,7,7,512) → (25088,)
        features = features.flatten()

        features_list.append(features)
        filenames_list.append(filename)

        print(f"✔ Saved features for {filename}")

    except Exception as e:
        print(f"❌ Error with {filename}: {e}")

# Convert to NumPy arrays
X = np.array(features_list)
image_names = np.array(filenames_list)

# Save to disk
np.save(FEATURES_OUTPUT, X)
np.save(FILENAMES_OUTPUT, image_names)

print("\n🎉 Feature extraction completed")
print(f"Total images processed: {len(image_names)}")
print(f"Saved features shape: {X.shape}")

import os
from PIL import Image

# -------- CONFIG --------
INPUT_DIR = "dataset"
OUTPUT_DIR = "cleaned_dataset"
IMAGE_SIZE = (224, 224)

# -----------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

count = 0

for img_name in os.listdir(INPUT_DIR):
    img_path = os.path.join(INPUT_DIR, img_name)

    # Only process image files
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")      # VGG16 requires RGB
            img = img.resize(IMAGE_SIZE) # 224x224
            img.save(os.path.join(OUTPUT_DIR, img_name))
            count += 1

    except Exception as e:
        print(f"❌ Skipping corrupted image: {img_name}")

print(f"✅ Cleaning completed! {count} images processed.")

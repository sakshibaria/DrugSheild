import os
from PIL import Image
import imagehash

INPUT_DIR = "cleaned_dataset"
OUTPUT_DIR = "deduplicated_dataset"
HASH_THRESHOLD = 5  # lower = stricter

os.makedirs(OUTPUT_DIR, exist_ok=True)

hashes = {}
kept = 0
removed = 0

for img_name in os.listdir(INPUT_DIR):
    img_path = os.path.join(INPUT_DIR, img_name)

    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    try:
        with Image.open(img_path) as img:
            img_hash = imagehash.phash(img)

        duplicate_found = False

        for existing_hash in hashes:
            if abs(img_hash - existing_hash) <= HASH_THRESHOLD:
                duplicate_found = True
                removed += 1
                break

        if not duplicate_found:
            hashes[img_hash] = img_name
            img.save(os.path.join(OUTPUT_DIR, img_name))
            kept += 1

    except Exception:
        print(f"❌ Error processing {img_name}")

print("✅ Duplicate removal completed!")
print(f"🟢 Images kept: {kept}")
print(f"🔴 Duplicates removed: {removed}")

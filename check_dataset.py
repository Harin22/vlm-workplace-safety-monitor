import json
import os

DATASET_FILE = "dataset/train.json"

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Training examples: {len(data)}")

for i, item in enumerate(data, 1):
    image_path = os.path.join("dataset", item["image"])

    if os.path.exists(image_path):
        status = "OK"
    else:
        status = "MISSING"

    print(f"{i}. {item['image']} -> {status}")

print("\ndataset check done..")
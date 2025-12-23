import os
import shutil
import random

# Input folders (your current output)
IMAGES_FOLDER = "images"
LABELS_FOLDER = "labels"

# Output dataset structure
DATASET_DIR = "dataset"
TRAIN_RATIO = 0.8   # 80% train, 20% val

# Make directories
dirs = [
    f"{DATASET_DIR}/images/train",
    f"{DATASET_DIR}/images/val",
    f"{DATASET_DIR}/labels/train",
    f"{DATASET_DIR}/labels/val",
]

for d in dirs:
    os.makedirs(d, exist_ok=True)

# List all images
images = sorted([f for f in os.listdir(IMAGES_FOLDER)
                if f.endswith(".jpg") or f.endswith(".png")])
random.shuffle(images)

train_count = int(len(images) * TRAIN_RATIO)

train_images = images[:train_count]
val_images = images[train_count:]


def move_files(file_list, subset):
    for img_file in file_list:
        label_file = img_file.replace(".jpg", ".txt").replace(".png", ".txt")

        shutil.copy(os.path.join(IMAGES_FOLDER, img_file),
                    os.path.join(DATASET_DIR, f"images/{subset}", img_file))

        shutil.copy(os.path.join(LABELS_FOLDER, label_file),
                    os.path.join(DATASET_DIR, f"labels/{subset}", label_file))


# Move train files
move_files(train_images, "train")

# Move val files
move_files(val_images, "val")

print(f"Dataset prepared successfully!")
print(f"Train images: {len(train_images)}")
print(f"Val images: {len(val_images)}")

# Create dataset.yaml
yaml_content = """train: dataset/images/train
val: dataset/images/val

nc: 5
names: ["car", "bus", "jeep", "truck", "ambulance"]
"""
with open("dataset.yaml", "w") as f:
    f.write(yaml_content)


print("dataset.yaml created!")

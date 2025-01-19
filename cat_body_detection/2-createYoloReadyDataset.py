import os
import random
import shutil
from PIL import Image
import glob

# Paths
dataset_path = "./cat_body_detection/datasets/Image Dataset"
yolo_dataset_path = "./cat_body_detection/datasets/yolo_dataset"
os.makedirs(yolo_dataset_path, exist_ok=True)

# Create YOLO structure
for split in ["train", "validation"]:
    for subfolder in ["images", "labels"]:
        os.makedirs(os.path.join(yolo_dataset_path, split, subfolder), exist_ok=True)

# Get cat images (assumes all cat images are in "1 Cats" with no subdirectories)
cat_images = [
    os.path.join(dataset_path, "1 Cats", f)
    for f in os.listdir(os.path.join(dataset_path, "1 Cats"))
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]


# Get random images recursively from "2 Random Images" and all its subfolders
random_images = glob.glob(os.path.join(dataset_path, "2 Random Images", "**", "*.*"), recursive=True)
random_images = [img for img in random_images if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Split into train and validation (80-20 split)
random.shuffle(cat_images)
random.shuffle(random_images)
train_cats = cat_images[:int(0.8 * len(cat_images))]
val_cats = cat_images[int(0.8 * len(cat_images)):]
train_randoms = random_images[:int(0.8 * len(random_images))]
val_randoms = random_images[int(0.8 * len(random_images)):]

# Function to create dummy YOLO labels
def create_yolo_label(image_path, label_path, label_id=0):
    with Image.open(image_path) as img:
        width, height = img.size
    # Dummy bounding box (covering the entire image)
    yolo_bbox = f"{label_id} 0.5 0.5 1.0 1.0"  # ClassID, X_center, Y_center, Width, Height (normalized)
    with open(label_path, "w") as f:
        f.write(yolo_bbox)

# Process images and labels
def process_images(image_paths, split, has_label):
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        # Copy image to images folder
        dest_image_path = os.path.join(yolo_dataset_path, split, "images", filename)
        shutil.copy(image_path, dest_image_path)

        # Create label file
        dest_label_path = os.path.join(yolo_dataset_path, split, "labels", filename.replace(os.path.splitext(filename)[1], ".txt"))
        if has_label:
            create_yolo_label(image_path, dest_label_path, label_id=0)  # Label "0" for "cat"
        else:
            # Create an empty label file for non-cat images
            open(dest_label_path, "w").close() 

# Generate YOLO dataset
process_images(train_cats, "train", has_label=True)
process_images(val_cats, "validation", has_label=True)
process_images(train_randoms, "train", has_label=False)
process_images(val_randoms, "validation", has_label=False)

print(f"Dataset organized at {yolo_dataset_path}")

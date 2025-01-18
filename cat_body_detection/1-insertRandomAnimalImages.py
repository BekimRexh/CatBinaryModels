import fiftyone as fo
import fiftyone.zoo as foz
import os
from fiftyone import ViewField as F

# Define target classes (make sure these match exactly with the dataset)
target_classes = ["Dog", "Elephant", "Lion", "Horse", "Rabbit"]

# Number of images per class
images_per_class = 100

# Desired directory to save the exported dataset
dataset_dir = r"C:\Users\bekim\OneDrive\Documents\VS Code\CatAppBinaryCnn\cat_face_detection\Image Dataset\More random_images\animal_images"

# Clear the directory if needed
if os.path.exists(dataset_dir):
    import shutil
    shutil.rmtree(dataset_dir)
os.makedirs(dataset_dir, exist_ok=True)

# Load the Open Images V6 dataset with max samples for all classes
dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="train",  # Use 'train' for more images
    label_types=["detections"],
    classes=target_classes,
    max_samples=images_per_class * len(target_classes),
    dataset_name="open-images-v6-custom",
    shuffle=True,
)

print("Dataset summary:")
print(dataset)

# Filter dataset to ensure 100 images per class
filtered_samples = []
for cls in target_classes:
    # Filter samples that have at least one detection with label == cls
    class_view = dataset.filter_labels("ground_truth", F("label") == cls)
    class_samples = class_view.take(images_per_class)
    if len(class_samples) < images_per_class:
        print(f"Warning: Only found {len(class_samples)} samples for class '{cls}'")
    filtered_samples.extend(class_samples)

if not filtered_samples:
    print("No samples found for the specified classes. Please verify your class names, dataset split, or reduce images_per_class.")
    exit()

# Create a new dataset with the filtered samples
filtered_dataset = fo.Dataset("filtered-open-images-v6")
for sample in filtered_samples:
    filtered_dataset.add_sample(sample)

# Export the filtered dataset to your desired directory
filtered_dataset.export(
    export_dir=dataset_dir,
    dataset_type=fo.types.FiftyOneImageDetectionDataset,
    label_field="ground_truth",
)

print(f"Filtered dataset exported to {dataset_dir}")

# Launch the FiftyOne App to visualize the dataset
session = fo.launch_app(filtered_dataset)
print(filtered_dataset)

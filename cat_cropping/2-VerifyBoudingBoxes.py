import tensorflow as tf
import os
import cv2 as cv
import glob

def draw_bounding_boxes_and_save(csv_file, image_dir, output_dir, image_size=(512, 512)):
    """
    Load dataset, draw bounding boxes, and save images with bounding boxes.
    
    Args:
        csv_file (str): Path to the CSV file containing bounding box data.
        image_dir (str): Directory where the original images are stored.
        output_dir (str): Directory to save the images with bounding boxes.
        image_size (tuple): Target image size (width, height).
    """
    # Prepare image paths and bounding boxes using the original function logic
    paths = []
    bboxes = []
    image_files = {
        os.path.basename(f): f.replace('\\', '/')
        for f in glob.glob(os.path.join(image_dir, '**', '*.*'), recursive=True)
    }

    with open(csv_file, 'r') as file:
        lines = file.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split(',')
            img_filename = parts[0]
            try:
                bbox = list(map(float, parts[1:]))
                if len(bbox) != 4:
                    print(f"Skipping invalid bounding box: {bbox}")
                    continue
                img_path = image_files.get(img_filename)
                if img_path and os.path.exists(img_path):
                    paths.append(img_path)
                    bboxes.append(bbox)
                else:
                    print(f"File does not exist: {img_filename}")
            except ValueError:
                print(f"Error parsing line: {line.strip()}")

    # Debug: Check dataset size
    print(f"Number of valid images: {len(paths)}")
    print(f"Number of valid bounding boxes: {len(bboxes)}")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each image and save with bounding box
    for img_path, bbox in zip(paths, bboxes):
        # Read the image
        image = cv.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue

        # Resize the image (optional)
        resized_image = cv.resize(image, image_size)

        # Extract bounding box values and scale them for resized image
        x_min, y_min, width, height = map(int, bbox)
        x_max = x_min + width
        y_max = y_min + height

        # Scale bounding box to new image size
        h_scale = image_size[1] / image.shape[0]
        w_scale = image_size[0] / image.shape[1]
        x_min = int(x_min * w_scale)
        y_min = int(y_min * h_scale)
        x_max = int(x_max * w_scale)
        y_max = int(y_max * h_scale)

        # Draw the bounding box
        cv.rectangle(resized_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Save the image with bounding box
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        cv.imwrite(output_path, resized_image)

    print(f"Processed images saved to: {output_dir}")


# Paths to CSVs and image directories
train_csv = './cat_face_detection/Cat Images/train_bounding_boxes.csv'
validation_csv = './cat_face_detection/Cat Images/validation_bounding_boxes.csv'
train_image_dir = './cat_face_detection/Cat Images/train'
validation_image_dir = './cat_face_detection/Cat Images/validation'

# Output directories
output_train_dir = './cat_face_detection/Cats Boxed/train'
output_validation_dir = './cat_face_detection/Cats Boxed/validation'

# Process training and validation datasets
draw_bounding_boxes_and_save(train_csv, train_image_dir, output_train_dir)
draw_bounding_boxes_and_save(validation_csv, validation_image_dir, output_validation_dir)

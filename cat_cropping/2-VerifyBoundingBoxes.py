import cv2
import os
import csv
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Paths
model_path = 'yolov5s.pt'  # Pre-trained YOLOv5 model (downloaded automatically)
train_cats_dir = './cat_face_detection/Cat Images/train'
validation_cats_dir = './cat_face_detection/Cat Images/validation'
output_train_csv = './cat_face_detection/Cat Images/train_bounding_boxes.csv'
output_validation_csv = './cat_face_detection/Cat Images/validation_bounding_boxes.csv'

# Initialize YOLO Model
model = YOLO(model_path)  # Load the pre-trained YOLOv5 model

def process_folder_with_yolo(image_dir, output_csv, class_label="cat"):
    """
    Detect objects (e.g., cat faces) in images using YOLO and save bounding boxes to CSV.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'x_min', 'y_min', 'width', 'height'])  # CSV header

        for subdir, _, files in os.walk(image_dir):
            for img_file in files:
                if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):  # Check for image files
                    img_path = os.path.join(subdir, img_file)
                    results = model(img_path, verbose=False)  # Run YOLO detection
                    
                    for result in results:
                        for box in result.boxes.xywh:  # Get bounding boxes (x_center, y_center, w, h)
                            x_center, y_center, width, height = box.tolist()
                            x_min = x_center - width / 2
                            y_min = y_center - height / 2

                            # Convert to integers for CSV storage
                            x_min = int(x_min)
                            y_min = int(y_min)
                            width = int(width)
                            height = int(height)

                            # Write the bounding box to the CSV
                            writer.writerow([img_file, x_min, y_min, width, height])

    print(f"Bounding box data saved to {output_csv}")


# Process train and validation folders
process_folder_with_yolo(train_cats_dir, output_train_csv)
process_folder_with_yolo(validation_cats_dir, output_validation_csv)


def visualize_yolo_predictions(image_path, model):
    """
    Visualize YOLO predictions on an image.
    """
    results = model(image_path, verbose=False)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for result in results:
        for box in result.boxes.xywh:
            x_center, y_center, width, height = box.tolist()
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            width = int(width)
            height = int(height)

            # Draw bounding box on the image
            cv2.rectangle(img, (x_min, y_min), (x_min + width, y_min + height), (255, 0, 0), 2)

    # Display the image
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Example usage
visualize_yolo_predictions('./cat_face_detection/Cat Images/sample.jpg', model)

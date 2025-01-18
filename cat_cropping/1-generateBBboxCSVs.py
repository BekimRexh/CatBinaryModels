import cv2 as cv
import os
import csv

# Paths
haar_cascade_path = 'cat_face_detection/haarcascade_frontalcatface.xml'
train_cats_dir = './cat_face_detection/Cat Images/images/train'
validation_cats_dir = './cat_face_detection/Cat Images/images/validation'
output_train_csv = './cat_face_detection/Cat Images/train_bounding_boxes.csv'
output_validation_csv = './cat_face_detection/Cat Images/validation_bounding_boxes.csv'

# Fixed image size
fixed_image_size = (224, 224)

# Initialize Haar Cascade
cascade = cv.CascadeClassifier(haar_cascade_path)

def process_folder(image_dir, output_csv):
    """
    Process a folder with cat images, resize images, detect faces,
    adjust bounding boxes, and write to CSV.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'x_min', 'y_min', 'width', 'height'])  # CSV header

        for subdir, _, files in os.walk(image_dir):
            for img_file in files:
                if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):  # Check for image files
                    img_path = os.path.join(subdir, img_file)
                    image = cv.imread(img_path)

                    if image is not None:
                        original_height, original_width = image.shape[:2]

                        # Detect faces on original image
                        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                        faces = cascade.detectMultiScale(
                            gray_image,
                            scaleFactor=1.1,
                            minNeighbors=3,
                            minSize=(30, 30)
                        )

                        # Resize image to fixed size
                        resized_image = cv.resize(image, fixed_image_size, interpolation=cv.INTER_AREA)

                        # Calculate scaling factors
                        scale_x = fixed_image_size[0] / original_width
                        scale_y = fixed_image_size[1] / original_height

                        # Adjust bounding boxes
                        for (x, y, w, h) in faces:
                            x_new = int(x * scale_x)
                            y_new = int(y * scale_y)
                            w_new = int(w * scale_x)
                            h_new = int(h * scale_y)
                            writer.writerow([img_file, x_new, y_new, w_new, h_new])

                        # Optionally save the resized image
                        # output_image_path = os.path.join('resized_images', img_file)
                        # os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
                        # cv.imwrite(output_image_path, resized_image)
                    else:
                        print(f"Failed to load image: {img_path}")

    print(f"Bounding box data saved to {output_csv}")

# Process train and validation "Cats" folders
process_folder(train_cats_dir, output_train_csv)
process_folder(validation_cats_dir, output_validation_csv)

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# Define the custom GIoU loss
def giou_loss(y_true, y_pred):
    x1_true, y1_true, x2_true, y2_true = tf.split(y_true, 4, axis=-1)
    x1_pred, y1_pred, x2_pred, y2_pred = tf.split(y_pred, 4, axis=-1)

    true_area = (x2_true - x1_true) * (y2_true - y1_true)
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)

    x1_int = tf.maximum(x1_true, x1_pred)
    y1_int = tf.maximum(y1_true, y1_pred)
    x2_int = tf.minimum(x2_true, x2_pred)
    y2_int = tf.minimum(y2_true, y2_pred)

    int_area = tf.maximum(0.0, x2_int - x1_int) * tf.maximum(0.0, y2_int - y1_int)
    union_area = true_area + pred_area - int_area

    iou = int_area / tf.maximum(union_area, tf.keras.backend.epsilon())

    x1_enc = tf.minimum(x1_true, x1_pred)
    y1_enc = tf.minimum(y1_true, y1_pred)
    x2_enc = tf.maximum(x2_true, x2_pred)
    y2_enc = tf.maximum(y2_true, y2_pred)

    enc_area = (x2_enc - x1_enc) * (y2_enc - y1_enc)

    giou = iou - (enc_area - union_area) / tf.maximum(enc_area, tf.keras.backend.epsilon())
    return 1 - giou

# Define the custom IoU metric
def iou_metric(y_true, y_pred):
    x1_true, y1_true, x2_true, y2_true = tf.split(y_true, 4, axis=-1)
    x1_pred, y1_pred, x2_pred, y2_pred = tf.split(y_pred, 4, axis=-1)

    x1_int = tf.maximum(x1_true, x1_pred)
    y1_int = tf.maximum(y1_true, y1_pred)
    x2_int = tf.minimum(x2_true, x2_pred)
    y2_int = tf.minimum(y2_true, y2_pred)

    int_area = tf.maximum(0.0, x2_int - x1_int) * tf.maximum(0.0, y2_int - y1_int)
    true_area = (x2_true - x1_true) * (y2_true - y1_true)
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)

    union_area = true_area + pred_area - int_area

    return tf.reduce_mean(int_area / tf.maximum(union_area, tf.keras.backend.epsilon()))

# Paths
model_path = './cat_face_detection/new_cat_face_detection_model.h5'
validation_dir = r'C:\Users\bekim\OneDrive\Documents\VS Code\CatAppBinaryCnn\cat_face_detection\Image Dataset\1 Cats'
output_dir = './cat_face_detection/bounding_box_predictions'
os.makedirs(output_dir, exist_ok=True)

# Load the trained model with custom object scope
with tf.keras.utils.custom_object_scope({'giou_loss': giou_loss, 'iou_metric': iou_metric}):
    model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

# Helper function to preprocess images
def preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image_resized = image.resize(target_size)
    image_array = np.array(image_resized) / 255.0  # Normalize to [0, 1]
    return image_array, original_size, image

# Helper function to draw bounding boxes
def draw_bounding_box(image, bbox, original_size):
    draw = ImageDraw.Draw(image)

    # Denormalize bounding box coordinates
    original_width, original_height = original_size
    x_min = int(bbox[0] * original_width)
    y_min = int(bbox[1] * original_height)
    x_max = int((bbox[0] + bbox[2]) * original_width)
    y_max = int((bbox[1] + bbox[3]) * original_height)

    # Calculate box width and height
    box_width = x_max - x_min
    box_height = y_max - y_min

    # **Sliding scale based on relative box size**
    box_ratio = box_width / original_width
    right_increase_factor = 0.6 + (1 - box_ratio) * 0.6  # Base increase is 60%, max increase is 120%

    # **Apply transformations**
    x_min = int(x_min - 0.2 * box_width)  # Expand left by 20%
    x_max = int(x_max + right_increase_factor * box_width)  # Increase right side using sliding scale
    y_min = int(y_min - 0.2 * box_height)  # Expand top by 20%
    y_max = int(y_max + 0.2 * box_height)  # Expand bottom by 20%

    print(f"Extended BBox (denormalized): [{x_min}, {y_min}, {x_max}, {y_max}] for image size {original_size}")

    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
    return image

# Function to process and predict bounding box on a single image
def test_on_single_image(image_path, save_output=True):
    print(f"Processing image: {image_path}")

    if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise ValueError("File must be an image with .png, .jpg, or .jpeg extension.")

    img_array, original_size, original_image = preprocess_image(image_path)

    prediction = model.predict(np.expand_dims(img_array, axis=0))[0]
    print(f"Prediction (normalized): {prediction}")

    output_image = draw_bounding_box(original_image.copy(), prediction, original_size)

    if save_output:
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        output_image.save(output_path)
        print(f"Processed {image_path}, saved to {output_path}")

    return output_image


# Function to process all images in the validation directory
def process_validation_images():
    print("Processing validation images...")
    for file_name in os.listdir(validation_dir)[:50]:
        file_path = os.path.join(validation_dir, file_name)

        if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_array, original_size, original_image = preprocess_image(file_path)

        prediction = model.predict(np.expand_dims(img_array, axis=0))[0]
        print(f"Prediction (normalized): {prediction}")

        output_image = draw_bounding_box(original_image.copy(), prediction, original_size)

        output_path = os.path.join(output_dir, file_name)
        output_image.save(output_path)

        print(f"Processed {file_name}, saved to {output_path}")

    print(f"Bounding box predictions saved to: {output_dir}")



# Example usage:
# To test on one image:
test_on_single_image('cat_face_detection/sample4.jpg')

# To process all images in the validation directory:
# process_validation_images()

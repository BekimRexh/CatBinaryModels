import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
model_path = 'C:/Users/bekim/OneDrive/Documents/VS Code/CatAppBinaryCnn/cat_body_detection/finalCatObjectModel/weights/cat_object_detector_2'
model = tf.saved_model.load(model_path)

def preprocess_image(image_path, input_size=(640, 640)):
    """
    Load and preprocess the image for the model.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize(input_size)
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    input_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    input_tensor = tf.expand_dims(input_tensor, axis=0)  # Add batch dimension
    return input_tensor

def decode_predictions(predictions, confidence_threshold=0.5):
    """
    Decode YOLOv8 predictions for binary detection (object vs. no object).
    """
    predictions = tf.squeeze(predictions, axis=0)

    # Debugging: Check predictions shape and values
    print("Predictions shape:", predictions.shape)
    print("Confidence values (sample):", predictions[4, :10].numpy())

    # Extract confidence scores
    confidences = predictions[4, :]  # Confidence scores
    confident_indices = tf.where(confidences > confidence_threshold)

    print("Confident indices:", confident_indices.numpy())

    if tf.shape(confident_indices)[0] == 0:
        return "No, object not found"

    # Iterate through confident detections and print bounding boxes
    for idx in tf.squeeze(confident_indices):
        confidence = confidences[idx]
        bounding_box = predictions[:4, idx]
        print(f"Confidence: {confidence.numpy()}, Bounding box: {bounding_box.numpy()}")
        return "Yes, object detected"

    return "No, object not found"


# Main function
def main():
    # Path to your test image
    test_image_path = 'cat_body_detection\sample.png'  # Replace with the path to your image

    # Preprocess the image
    input_tensor = preprocess_image(test_image_path)

    # Run predictions
    outputs = model(input_tensor)

    # Decode predictions
    result = decode_predictions(outputs, confidence_threshold=0.5)  # Removed cat_class_id
    print(result)


if __name__ == "__main__":
    main()

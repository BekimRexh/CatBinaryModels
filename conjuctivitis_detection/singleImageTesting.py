import tensorflow as tf
from keras_preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('finalConjunctModel.h5')

# Path to your random cat image
image_path = 'C:/Users/bekim/OneDrive/Documents/VS Code/CatAppCnnTest/sample.jpg'  # Replace with the path to your test image

# Preprocess the image (same preprocessing as training)
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))  # Resize the image to match training size
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize (same as during training)
    return img_array

# Preprocess the image
processed_image = preprocess_image(image_path)

# Make a prediction
prediction = model.predict(processed_image)

# Interpret the result
if prediction[0] > 0.5:
    print("The model predicts: Likely Conjunctivitis")
else:
    print("The model predicts: Healthy")

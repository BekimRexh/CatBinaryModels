import tensorflow as tf

# Load the YOLO TensorFlow SavedModel
saved_model_path = 'C:/Users/bekim/OneDrive/Documents/VS Code/CatBinaryModels/cat_body_detection/cat_detector3/weights/best_saved_model'
model = tf.saved_model.load(saved_model_path)

# Define a Keras-compatible model wrapper
class YoloKerasModel(tf.keras.Model):
    def __init__(self, model):
        super(YoloKerasModel, self).__init__()
        self.model = model

    def call(self, inputs):
        return self.model(inputs)
    
    def get_config(self):
        # Return an empty config or just skip this method entirely
        return {}



# Wrap the YOLO model
keras_model = YoloKerasModel(model)

# Define input shape (e.g., 640x640x3 for YOLOv8)
input_shape = (1, 640, 640, 3)

# Build the model by calling it on dummy data
dummy_input = tf.random.normal(input_shape)
_ = keras_model(dummy_input)  # Perform a forward pass to initialize the model

# Save the whole model in TensorFlow SavedModel format
keras_model.save('C:/Users/bekim/OneDrive/Documents/VS Code/CatBinaryModels/cat_body_detection/cat_object_detector_3', save_format='tf')

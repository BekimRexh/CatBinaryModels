import tensorflow as tf
import os
import csv
import random
import glob
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib


print("TensorFlow version:", tf.__version__)
print("Built with GPU support:", tf.test.is_built_with_cuda())
print("GPUs available:", tf.config.list_physical_devices('GPU'))

"""
Cat Face Bounding Box Detection and Prediction

This script trains a bounding box prediction model for detecting cat faces. The workflow involves:
1. Using `haarcascade_frontalcatface.xml` to obtain bounding box data for cat images.
2. Storing the bounding box data in CSV files, with each row containing the image filename and bounding box coordinates.
3. Training a MobileNetV2-based deep learning model to predict bounding boxes on unseen images of cats.

Steps:
- Load datasets with bounding box annotations from CSV files.
- Normalize bounding box coordinates based on image dimensions.
- Train a neural network to predict bounding boxes from images.
- Evaluate the model using IoU (Intersection over Union) and GIoU loss.
- Save the trained model and visualize predictions for validation.
"""



def load_dataset_with_bounding_boxes(csv_file, image_dir, image_size=(224, 224)):
    """
    Load dataset with normalized bounding boxes from a CSV file.
    """
    paths = []
    bboxes = []

    image_files = {
        os.path.basename(f): f.replace('\\', '/')
        for f in glob.glob(os.path.join(image_dir, '**', '*.*'), recursive=True)
    }

    with open(csv_file, 'r') as file:
        lines = file.readlines()[1:]
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
                    # Normalize bounding box values by image size
                    x_min, y_min, width, height = bbox
                    normalized_bbox = [
                        x_min / image_size[0],
                        y_min / image_size[1],
                        width / image_size[0],
                        height / image_size[1],
                    ]
                    bboxes.append(normalized_bbox)
                else:
                    print(f"File does not exist: {img_filename}")
            except ValueError:
                print(f"Error parsing line: {line.strip()}")

    print("Number of valid images:", len(paths))
    print("Number of valid bounding boxes:", len(bboxes))
    print("Normalized Bounding Boxes Sample:", bboxes[:5])


    if len(paths) == 0 or len(bboxes) == 0:
        raise ValueError("No valid images or bounding boxes found.")

    paths_ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(paths, dtype=tf.string))
    bboxes_ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(bboxes, dtype=tf.float32))

    dataset = tf.data.Dataset.zip((paths_ds, bboxes_ds))

    def process_entry(img_path, bbox):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)  
        img = tf.image.resize(img, (224, 224)) / 255.0  
        img = tf.ensure_shape(img, (224, 224, 3))  
        bbox = tf.reshape(bbox, [4]) 
        return img, bbox

    dataset = dataset.map(process_entry)
    return dataset

def giou_loss(y_true, y_pred):
    """
    Generalized Intersection over Union (GIoU) Loss:
    - Measures the overlap between true and predicted bounding boxes.
    - Penalizes predictions that do not overlap well with the ground truth.
    """
    x1_true, y1_true, w_true, h_true = tf.split(y_true, 4, axis=-1)
    x1_pred, y1_pred, w_pred, h_pred = tf.split(y_pred, 4, axis=-1)

    x2_true, y2_true = x1_true + w_true, y1_true + h_true
    x2_pred, y2_pred = x1_pred + w_pred, y1_pred + h_pred

    x1_inter = tf.maximum(x1_true, x1_pred)
    y1_inter = tf.maximum(y1_true, y1_pred)
    x2_inter = tf.minimum(x2_true, x2_pred)
    y2_inter = tf.minimum(y2_true, y2_pred)

    inter_area = tf.maximum(0.0, x2_inter - x1_inter) * tf.maximum(0.0, y2_inter - y1_inter)
    true_area = w_true * h_true
    pred_area = w_pred * h_pred
    union_area = true_area + pred_area - inter_area

    iou = inter_area / tf.maximum(union_area, 1e-6)

    x1_enclosing = tf.minimum(x1_true, x1_pred)
    y1_enclosing = tf.minimum(y1_true, y1_pred)
    x2_enclosing = tf.maximum(x2_true, x2_pred)
    y2_enclosing = tf.maximum(y2_true, y2_pred)

    enclosing_area = tf.maximum(0.0, x2_enclosing - x1_enclosing) * tf.maximum(0.0, y2_enclosing - y1_enclosing)
    giou = iou - (enclosing_area - union_area) / tf.maximum(enclosing_area, 1e-6)
    return 1.0 - giou

def iou_metric(y_true, y_pred):
    """
    Calculate Intersection over Union (IoU) for bounding boxes.
    """
    x1_true, y1_true, w_true, h_true = tf.split(y_true, 4, axis=-1)
    x1_pred, y1_pred, w_pred, h_pred = tf.split(y_pred, 4, axis=-1)
    
    x2_true, y2_true = x1_true + w_true, y1_true + h_true
    x2_pred, y2_pred = x1_pred + w_pred, y1_pred + h_pred
    
    x1_inter = tf.maximum(x1_true, x1_pred)
    y1_inter = tf.maximum(y1_true, y1_pred)
    x2_inter = tf.minimum(x2_true, x2_pred)
    y2_inter = tf.minimum(y2_true, y2_pred)
    
    inter_area = tf.maximum(0.0, x2_inter - x1_inter) * tf.maximum(0.0, y2_inter - y1_inter)
    
    true_area = w_true * h_true
    pred_area = w_pred * h_pred
    union_area = true_area + pred_area - inter_area
    
    iou = inter_area / tf.maximum(union_area, 1e-6)
    return tf.reduce_mean(iou)


def create_bounding_box_model(input_shape=(224, 224, 3)):
    """
    Create a bounding box prediction model:
    - Uses MobileNetV2 as the base for feature extraction.
    - Fine-tunes the last 50 layers for task-specific training.
    - Outputs 4 values (x_min, y_min, width, height) as normalized coordinates.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights='imagenet'
    )

    base_model.trainable = True  

    for layer in base_model.layers[:-50]: 
        layer.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(224, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(4, activation='sigmoid')
    ])
    return model


train_csv = './cat_face_detection/Cat Images/train_bounding_boxes.csv'
validation_csv = './cat_face_detection/Cat Images/validation_bounding_boxes.csv'
train_image_dir = './cat_face_detection/Cat Images/images/train'
validation_image_dir = './cat_face_detection/Cat Images/images/validation'


train_dataset = load_dataset_with_bounding_boxes(train_csv, train_image_dir)
validation_dataset = load_dataset_with_bounding_boxes(validation_csv, validation_image_dir)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=1000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model = create_bounding_box_model()

model.compile(
    optimizer=optimizer,
    loss=giou_loss,
    metrics=[iou_metric]
)

history = model.fit(
    train_dataset.batch(32).prefetch(tf.data.AUTOTUNE),
    validation_data=validation_dataset.batch(32).prefetch(tf.data.AUTOTUNE),
    epochs=30,
    callbacks=[]
)

def visualize_predictions(image, y_true, y_pred, image_size=(224, 224)):
    """
    Visualize ground truth and predicted bounding boxes on an image.
    - Green box: Ground truth bounding box.
    - Red box: Predicted bounding box.
    """
    img = image.numpy()
    img = (img * 255).astype('uint8') 

    x1_true, y1_true, w_true, h_true = y_true * image_size[0], y_true * image_size[1], y_true * image_size[0], y_true * image_size[1]
    x1_pred, y1_pred, w_pred, h_pred = y_pred * image_size[0], y_pred * image_size[1], y_pred * image_size[0], y_pred * image_size[1]

    plt.imshow(img)
    plt.gca().add_patch(plt.Rectangle((x1_true, y1_true), w_true, h_true, fill=False, edgecolor='green', linewidth=2))
    plt.gca().add_patch(plt.Rectangle((x1_pred, y1_pred), w_pred, h_pred, fill=False, edgecolor='red', linewidth=2))
    plt.show()


model.save('./cat_face_detection/new_cat_face_detection_model.h5')
model.export('./cat_face_detection/no_format_cat_face_model')


for img_batch, bbox_batch in train_dataset.batch(32).take(1):
    print("Image batch shape:", img_batch.shape)  
    print("Bounding box batch shape:", bbox_batch.shape)  


for img_batch, bbox_batch in validation_dataset.batch(32).take(1):
    img_batch = tf.ensure_shape(img_batch, (None, 224, 224, 3)) 
    pred_bboxes = model.predict(img_batch)

    for i in range(len(img_batch)):
        visualize_predictions(img_batch[i], bbox_batch[i].numpy(), pred_bboxes[i], image_size=(224, 224))






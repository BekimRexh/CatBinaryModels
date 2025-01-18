import os
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import pandas as pd

# Paths
dataset_path = "./cat_face_detection/Image Dataset"
model_save_path = "./cat_face_detection/cat_face_detector_mobilenetv2.h5"
output_dir = "./cat_face_detection/output"
os.makedirs(output_dir, exist_ok=True)

# Load datasets using `tf.data`
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(dataset_path, 'train'),
    labels='inferred',
    label_mode='binary',
    image_size=(128, 128),
    batch_size=32,
    seed=123,
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(dataset_path, 'validation'),
    labels='inferred',
    label_mode='binary',
    image_size=(128, 128),
    batch_size=32,
    seed=123,
)

class_names = train_dataset.class_names
print("Class Names:", class_names)

# Compute class weights
labels_list = []
for _, labels in train_dataset:
    labels_list.extend(labels.numpy().astype(int))

class_weights = {
    0: len(labels_list) / np.sum(np.array(labels_list) == 0),
    1: len(labels_list) / np.sum(np.array(labels_list) == 1),
}
print("Class Weights:", class_weights)

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),              # Random horizontal flip
    tf.keras.layers.RandomRotation(0.1),                   # Random rotation (±10%)
    tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1),  # Random zoom (±10%)
                      width_factor=(-0.1, 0.1)),
    tf.keras.layers.RandomTranslation(height_factor=0.1,   # Random translation (±10%)
                             width_factor=0.1),
    tf.keras.layers.RandomBrightness(factor=0.15),          # Random brightness adjustment
])

# Normalize data
def preprocess_dataset(dataset, augment=False):
    def normalize(images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        return images, labels

    dataset = dataset.map(normalize)
    if augment:
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

train_dataset = preprocess_dataset(train_dataset, augment=True)
validation_dataset = preprocess_dataset(validation_dataset)

# Load MobileNetV2 Base Model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,  # Remove the classification head
    weights='imagenet'  # Pretrained weights
)

base_model.trainable = True  # Unfreeze the entire base model

# Freeze earlier layers
for layer in base_model.layers[:-30]:  # Tune this value for performance
    layer.trainable = False

# Build the Model
model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Pooling layer
    layers.Dropout(0.3),              # Regularization
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# Train the Model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=30,
    class_weight=class_weights,
    callbacks=[keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)],
)

# Save the Model
model.export('./cat_face_detection/no_format_model_cat_face')
# Save the model
model.save('./cat_face_detection/cat_face_model.h5')


# Plot Training Performance
plt.figure(figsize=(12, 6))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

performance_plot_path = os.path.join(output_dir, "training_performance.png")
plt.savefig(performance_plot_path, dpi=300)
plt.show()

# Evaluate the Model
y_pred = []
y_true = []

for images, labels in validation_dataset:
    preds = model.predict(images)
    y_pred.extend((preds > 0.5).astype(int).flatten())
    y_true.extend(labels.numpy().astype(int))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp_labels = class_names
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
disp.plot(cmap=plt.cm.Blues)
confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(confusion_matrix_path, dpi=300)
plt.show()

# Classification Report
report = classification_report(y_true, y_pred, target_names=disp_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df)

# Save Classification Report
report_csv_path = os.path.join(output_dir, "classification_report.csv")
report_df.to_csv(report_csv_path)
print(f"Classification report saved to: {report_csv_path}")

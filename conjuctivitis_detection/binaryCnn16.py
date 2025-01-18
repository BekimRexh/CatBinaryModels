import os
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, regularizers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import numpy as np
from sklearn.utils import class_weight
from keras.api.callbacks import ReduceLROnPlateau
import seaborn as sns
import pandas as pd

"""
Binary Classification of Healthy vs. Non-Healthy Cats for Conjunctivitis Detection

This script trains a convolutional neural network (CNN) model to classify cat images as 
healthy or non-healthy, with the following workflow:
1. Perform binary classification (healthy vs. non-healthy) based on labeled datasets.
2. Augment and balance training data to address class imbalances.
3. Train the model using a Sequential CNN architecture and save the trained model.
4. Evaluate performance using accuracy, loss, confusion matrix, and classification report.
5. The app will use the model's inference score to determine the likelihood of conjunctivitis:
    for example:
   - ≤ 0.5: Healthy
   - 0.5–0.65: Low chance of conjunctivitis
   - 0.65–0.8: Moderate chance of conjunctivitis
   - > 0.8: High chance of conjunctivitis

Key Features:
- Class weighting to handle imbalanced datasets.
- Custom data augmentation for better generalization.
- Label smoothing to reduce overfitting.
- Visualization of metrics, confusion matrix, and classification report for evaluation.
"""


raw_train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    './non-augmented-dataset-4',
    labels='inferred',
    label_mode='binary',  # Binary classification
    validation_split=0.02,
    subset='training',
    seed=123,
    image_size=(64, 64),
    batch_size=5
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    './non-augmented-dataset-4',
    labels='inferred',
    label_mode='binary',
    validation_split=0.02,
    subset='validation',
    seed=123,
    image_size=(64, 64),
    batch_size=5
)

class_names = raw_train_dataset.class_names
print("Class names:", class_names)

labels_list = []
for images, labels in raw_train_dataset:
    labels_list.extend(labels.numpy().astype(int).flatten())

all_labels = np.array(labels_list, dtype=int)

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(all_labels),
    y=all_labels
)

# Calculate class weights to address imbalance in the dataset
class_weight_dict = {int(k): v for k, v in zip(np.unique(all_labels), class_weights)}
print("Class weights:", class_weight_dict)

def random_zoom(image):
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    channels = shape[2]

    zoom_factor = tf.random.uniform([], 0.9, 1.1)

    new_height = tf.cast(tf.cast(height, tf.float32) * zoom_factor, tf.int32)
    new_width = tf.cast(tf.cast(width, tf.float32) * zoom_factor, tf.int32)

    image = tf.image.resize(image, [new_height, new_width])
    image = tf.image.resize_with_crop_or_pad(image, height, width)

    return image

def apply_augmentation(images):
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_brightness(images, max_delta=0.2)
    images = tf.map_fn(random_zoom, images)

    return images


# Apply data augmentation to the non-healthy class (label = 1) to balance the dataset
# Augmentations include random flipping, brightness adjustments, and zooming
def augment_images(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    labels = tf.cast(labels, tf.int32)
    mask = tf.equal(labels, 1)
    mask = tf.reshape(mask, (-1, 1, 1, 1))
    augmented_images = apply_augmentation(images)
    images = tf.where(mask, augmented_images, images)

    return images, labels

train_dataset = raw_train_dataset.map(augment_images, num_parallel_calls=tf.data.AUTOTUNE)

def normalize_images(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels

validation_dataset = validation_dataset.map(normalize_images, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

model = models.Sequential([
    layers.InputLayer(input_shape=(64, 64, 3)),
    layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.15),
    metrics=['accuracy']
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=70,
    class_weight=class_weight_dict,
    callbacks=[lr_scheduler],
    verbose=1
)

model.save('binaryCnn16-model.h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

# Evaluate model performance using a confusion matrix and classification report
# Visualize training and validation accuracy/loss trends over epochs
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy', marker='o')
plt.plot(epochs_range, val_acc, label='Validation Accuracy', marker='x')
plt.legend(loc='lower right')
plt.title('Training vs Validation Accuracy (Imbalanced Dataset)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss', marker='o')
plt.plot(epochs_range, val_loss, label='Validation Loss', marker='x')
plt.legend(loc='upper right')
plt.title('Training vs Validation Loss (Imbalanced Dataset)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('training_validation_accuracy_loss.png', dpi=300)
plt.show()

y_pred = []
y_true = []

for images, labels in validation_dataset:
    preds = model.predict(images, verbose=0)
    y_pred.extend(preds)
    y_true.extend(labels.numpy())

y_pred_class = (np.array(y_pred) > 0.5).astype('int32').flatten()
y_true = np.array(y_true).astype('int32').flatten()

cm = confusion_matrix(y_true, y_pred_class)

disp_labels = class_names  

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Imbalanced Dataset)')

plt.savefig('confusion_matrix.png', dpi=300)

plt.show()

report = classification_report(y_true, y_pred_class, target_names=disp_labels, output_dict=True)

report_df = pd.DataFrame(report).transpose()

report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')

plt.figure(figsize=(8, 6))
sns.heatmap(report_df.iloc[:, :-1], annot=True, cmap='Blues', fmt='.2f', cbar=False)

plt.title('Classification Report (Imbalanced Dataset)')
plt.ylabel('Classes')
plt.xlabel('Metrics')

plt.savefig('classification_report.png', dpi=300, bbox_inches='tight')

plt.show()

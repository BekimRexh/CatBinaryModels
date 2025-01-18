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




# Load the datasets using tf.data API
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

# Get class names
class_names = raw_train_dataset.class_names
print("Class names:", class_names)
# Output should be: Class names: ['1 Healthy', '2 Unhealthy']

# Compute class weights before applying augmentation
labels_list = []
for images, labels in raw_train_dataset:
    labels_list.extend(labels.numpy().astype(int).flatten())  # Ensure labels are integers

all_labels = np.array(labels_list, dtype=int)  # Convert to integer array

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(all_labels),
    y=all_labels
)

# Ensure keys in class_weight_dict are integers
class_weight_dict = {int(k): v for k, v in zip(np.unique(all_labels), class_weights)}
print("Class weights:", class_weight_dict)

# Define random zoom function
def random_zoom(image):
    # Get image dimensions
    height, width, channels = image.shape

    # Generate a random zoom factor between 0.9 and 1.1
    zoom_factor = tf.random.uniform([], 0.9, 1.1)

    # Calculate new dimensions
    new_height = tf.cast(height * zoom_factor, tf.int32)
    new_width = tf.cast(width * zoom_factor, tf.int32)

    # Resize the image
    image = tf.image.resize(image, [new_height, new_width])

    # Resize back to original dimensions
    image = tf.image.resize_with_crop_or_pad(image, height, width)

    return image

# Define random shift function
def random_shift(image):
    # Get image dimensions
    height, width, channels = image.shape

    # Maximum shift in pixels (10% of image dimensions)
    max_shift_height = int(height * 0.1)
    max_shift_width = int(width * 0.1)

    # Generate random shifts
    shift_height = tf.random.uniform([], -max_shift_height, max_shift_height + 1, dtype=tf.int32)
    shift_width = tf.random.uniform([], -max_shift_width, max_shift_width + 1, dtype=tf.int32)

    # Pad the image
    padded_image = tf.image.pad_to_bounding_box(
        image,
        max_shift_height,
        max_shift_width,
        height + 2 * max_shift_height,
        width + 2 * max_shift_width
    )

    # Crop the image to achieve the shift
    offset_height = max_shift_height + shift_height
    offset_width = max_shift_width + shift_width
    image = tf.image.crop_to_bounding_box(
        padded_image,
        offset_height,
        offset_width,
        height,
        width
    )

    return image

# Updated apply_augmentation function
def apply_augmentation(image):
    # Random flip
    image = tf.image.random_flip_left_right(image)

    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.2)

    # Random zoom
    image = random_zoom(image)

    # Random shift
    # image = random_shift(image)

    # Clip image to valid range
    # image = tf.clip_by_value(image, 0.0, 1.0)

    return image

# Updated augment_images function
def augment_images(images, labels):
    # Normalize images
    images = tf.cast(images, tf.float32) / 255.0
    labels = tf.cast(labels, tf.int32)

    # Create a mask for labels equal to 1
    mask = tf.equal(labels, 1)

    # Apply augmentations only to images where labels == 1
    def augment_if_label_one(image, label):
        return tf.cond(
            tf.equal(label, 1),
            lambda: apply_augmentation(image),
            lambda: image
        )

    # Apply the function to each image-label pair
    augmented_images = tf.map_fn(
        lambda x: augment_if_label_one(x[0], x[1]),
        (images, labels),
        dtype=tf.float32
    )

    return augmented_images, labels

# Apply the augmentation to the training dataset
train_dataset = raw_train_dataset.map(augment_images, num_parallel_calls=tf.data.AUTOTUNE)

# Normalize validation dataset
def normalize_images(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

validation_dataset = validation_dataset.map(normalize_images, num_parallel_calls=tf.data.AUTOTUNE)

# Prefetching for performance
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Build the model
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

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Learning rate scheduler callback
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=70,
    class_weight=class_weight_dict,
    callbacks=[lr_scheduler],
    verbose=1
)

model.save('binaryCnn14-model.h5')

# Plotting the training and validation accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

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

# Save the plot to your local directory
plt.savefig('training_validation_accuracy_loss.png', dpi=300)

plt.show()

# Evaluate the Model with Validation Data
y_pred = []
y_true = []

for images, labels in validation_dataset:
    preds = model.predict(images, verbose=0)
    y_pred.extend(preds)
    y_true.extend(labels.numpy())

y_pred_class = (np.array(y_pred) > 0.5).astype('int32').flatten()
y_true = np.array(y_true).astype('int32').flatten()

# Compute confusion matrix and display results
cm = confusion_matrix(y_true, y_pred_class)

# Use class names for labels
disp_labels = class_names  # ['1 Healthy', '2 Unhealthy']

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Imbalanced Dataset)')

# Save the confusion matrix plot
plt.savefig('confusion_matrix.png', dpi=300)

plt.show()

# Generate classification report as a dictionary
report = classification_report(y_true, y_pred_class, target_names=disp_labels, output_dict=True)

# Convert the classification report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Exclude 'accuracy', 'macro avg', and 'weighted avg' if desired
report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')

# Plot the classification report as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(report_df.iloc[:, :-1], annot=True, cmap='Blues', fmt='.2f', cbar=False)

plt.title('Classification Report (Imbalanced Dataset)')
plt.ylabel('Classes')
plt.xlabel('Metrics')

# Save the classification report plot
plt.savefig('classification_report.png', dpi=300, bbox_inches='tight')

plt.show()

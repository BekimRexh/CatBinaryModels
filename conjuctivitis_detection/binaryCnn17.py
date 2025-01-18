import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import numpy as np
from sklearn.utils import class_weight
import seaborn as sns
import pandas as pd

print("TensorFlow version:", tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "not available")

# Load datasets
raw_train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    './conjuctivitis_detection/HD Dataset',
    labels='inferred',
    label_mode='binary',
    validation_split=0.10,
    subset='training',
    seed=123,
    image_size=(128, 128),
    batch_size=4
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    './conjuctivitis_detection/HD Dataset',
    labels='inferred',
    label_mode='binary',
    validation_split=0.10,
    subset='validation',
    seed=123,
    image_size=(128, 128),
    batch_size=4
)

# Get class names
class_names = raw_train_dataset.class_names
print("Class names:", class_names)

# Compute class weights
labels_list = [label.numpy().astype(int) for _, label in raw_train_dataset.unbatch()]
all_labels = np.concatenate(labels_list)

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(all_labels),
    y=all_labels
)

class_weight_dict = {int(k): v for k, v in zip(np.unique(all_labels), class_weights)}
print("Class weights:", class_weight_dict)

# Data Augmentation Layer
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1.0 / 255),                  # Normalize pixel values to [0, 1]
    tf.keras.layers.RandomFlip("horizontal"),              # Random horizontal flip
    # tf.keras.layers.RandomRotation(0.1),                   # Random rotation (±10%)
    # tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1),  # Random zoom (±10%)
    #                   width_factor=(-0.1, 0.1)),
    tf.keras.layers.RandomTranslation(height_factor=0.1,   # Random translation (±10%)
                             width_factor=0.1),
    # tf.keras.layers.RandomContrast(factor=0.2),            # Random contrast adjustment
    tf.keras.layers.RandomBrightness(factor=0.1),          # Random brightness adjustment
])

# Normalize validation dataset
def normalize_images(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels

# Apply augmentations to class 1 only in the training dataset
def augment_class_1_only(images, labels):
    # Expand labels to match the shape of images
    labels_expanded = tf.reshape(labels, (-1, 1, 1, 1))  # Expand to [batch_size, 1, 1, 1]
    labels_expanded = tf.cast(labels_expanded, tf.float32)  # Cast to float32 for compatibility

    # Create a mask for class 1
    mask = tf.equal(labels_expanded, 1.0)  # True for class 1, False otherwise

    # Apply augmentation only to class 1
    augmented_images = tf.where(
        mask,                       # Condition: True for class 1
        data_augmentation(images),  # Augmented images for class 1
        images                      # Original images for other classes
    )
    return augmented_images, labels

validation_dataset = validation_dataset.map(normalize_images, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = raw_train_dataset.map(augment_class_1_only, num_parallel_calls=tf.data.AUTOTUNE)

# Prefetch datasets for performance
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Build the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(128, 128, 3)),
    data_augmentation,  # Apply data augmentation as part of the model pipeline
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.15),
    metrics=['accuracy']
)

# Learning rate scheduler callback
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
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

# Save the model
model.save('./conjuctivitis_detection/binaryCnn17-model.h5')

# Plot training history
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
disp_labels = class_names  # ['1 Healthy', '2 Unhealthy']

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Imbalanced Dataset)')
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

# Generate classification report as a dictionary
report = classification_report(y_true, y_pred_class, target_names=disp_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')

# Plot the classification report as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(report_df.iloc[:, :-1], annot=True, cmap='Blues', fmt='.2f', cbar=False)
plt.title('Classification Report (Imbalanced Dataset)')
plt.ylabel('Classes')
plt.xlabel('Metrics')
plt.savefig('classification_report.png', dpi=300, bbox_inches='tight')
plt.show()

# Import necessary libraries
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import numpy as np
from sklearn.utils import class_weight
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
    For example:
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

# Load datasets with low validation split to maximize training data
# The validation split is set low due to the limited number of images available. This was for the final model export.
# The validation split in testing was altered many times to a higher value to analyse the confusion matrix before exporting a final model.
raw_train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    './conjuctivitis_detection/HD Dataset',
    labels='inferred',
    label_mode='binary',
    validation_split=0.2, # Low validation split to use most images for training, 
    subset='training',
    seed=123,
    image_size=(512, 512), # High resolution to detect subtle conjunctivitis symptoms
    batch_size=4
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    './conjuctivitis_detection/HD Dataset',
    labels='inferred',
    label_mode='binary',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(512, 512),
    batch_size=4
)

class_names = raw_train_dataset.class_names
print("Class names:", class_names)

# Compute class weights to address imbalance between healthy and non-healthy images
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
# Augmentations applied with specific ranges to preserve the key features of conjunctivitis:
# - RandomRotation: Cats can tilt their heads naturally so rotation at 0.3 is good for data variance.
# - RandomZoom: Kept low to ensure key features like the cat's eyes remain visible.
# - RandomTranslation: Minimal to avoid displacing the eyes of the cat's face off of the image.
# - RandomBrightness: To account for varying lighting conditions cats can be in upon capture by pet owners.

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),             
    tf.keras.layers.RandomRotation(0.3),                   
    tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1),  
                      width_factor=(-0.1, 0.1)),
    tf.keras.layers.RandomTranslation(height_factor=0.1,   
                             width_factor=0.1),
    tf.keras.layers.RandomBrightness(factor=0.3),         
])

# Normalize validation dataset for consistent input scaling
def normalize_images(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels

# Apply augmentations to the training dataset
# Augmentation is applied to all images to improve generalization and balance classes.
def augment_images(images, labels):
    augmented_images = data_augmentation(images)
    normalized_images = tf.cast(augmented_images, tf.float32) / 255.0
    return normalized_images, labels

validation_dataset = validation_dataset.map(normalize_images, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = raw_train_dataset.map(augment_images, num_parallel_calls=tf.data.AUTOTUNE)

train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


# MobileNetV2 was chosen for its lightweight architecture, ideal for mobile deployment.
# Layers are frozen to preserve pre-trained weights, with the final 30 layers made trainable.
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(512, 512, 3),
    include_top=False,  
    weights='imagenet'  
)

base_model.trainable = True

# Unfreeze last 30 layers to fine-tune the model for the dataset
# This allows the model to adapt to the specific features of cat eyes.
for layer in base_model.layers[:-30]: 
    layer.trainable = False

# Add custom classification layers
model = tf.keras.Sequential([
    base_model,  # MobileNetV2 feature extractor
    tf.keras.layers.GlobalAveragePooling2D(),  # Reduce dimensions while retaining features
    tf.keras.layers.Dropout(0.3),              # Regularization to prevent overfitting of data
    tf.keras.layers.Dense(128, activation='relu'),  
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')  # For Binary classification
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.15),
    metrics=['accuracy']
)

# Learning rate scheduler callback to adapt learning rate dynamically
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
    epochs=50,  
    class_weight=class_weight_dict,
    callbacks=[lr_scheduler],
    verbose=1
)

# Export the model without a format for compatibility with TensorFlow.js (used in Expo Go) 
# This was not needed later as ejected from expo and used tensorFlow Lite 
model.export('./conjuctivitis_detection/no_format_model')
model.save('./conjuctivitis_detection/mobilenet_augmented_model.h5')

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
plt.title('Training vs Validation Accuracy (Augmented Dataset)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss', marker='o')
plt.plot(epochs_range, val_loss, label='Validation Loss', marker='x')
plt.legend(loc='upper right')
plt.title('Training vs Validation Loss (Augmented Dataset)')
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
plt.title('Confusion Matrix (Augmented Dataset)')
plt.savefig('confusion_matrix_augmented.png', dpi=300)
plt.show()

# Generate classification report as a dictionary
report = classification_report(y_true, y_pred_class, target_names=disp_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')

# Plot the classification report as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(report_df.iloc[:, :-1], annot=True, cmap='Blues', fmt='.2f', cbar=False)
plt.title('Classification Report (Augmented Dataset)')
plt.ylabel('Classes')
plt.xlabel('Metrics')
plt.savefig('classification_report_augmented.png', dpi=300, bbox_inches='tight')
plt.show()

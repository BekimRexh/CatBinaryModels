import os
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras.api import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import numpy as np

##### SECOND ITERATION #####
##### ACCURACY 68% #####
##### BETTER BALANCE BETWEEN CLASSIFICATION, ACCURACY SIMILAR #####
##### NEW ADDED DATA AUGMENTATION ######
##### NEW INCREASED EPOCHS TO 15 ######

# Set the data directory
train_data_dir = './new-dataset'

# Create the data generator with data augmentation for training
datagen_with_aug = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    validation_split=0.1,  # Split off 10% of the data for validation
    rotation_range=20,  # Random rotation between 0 and 20 degrees
    width_shift_range=0.1,  # Random horizontal shift
    height_shift_range=0.1,  # Random vertical shift
    horizontal_flip=True,  # Randomly flip inputs horizontally
    fill_mode='nearest'  # Fill strategy for newly created pixels
)

# Training data generator with augmentation
train_generator_with_aug = datagen_with_aug.flow_from_directory(
    train_data_dir,
    target_size=(64, 64),  # Use smaller images for simplicity
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True
)

# Validation data generator (no augmentation)
datagen = ImageDataGenerator(
    rescale=1./255,  # Only rescale for validation
    validation_split=0.1
)

validation_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Define a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with augmented data
history_aug = model.fit(
    train_generator_with_aug,
    validation_data=validation_generator,
    epochs=15,  # Same number of epochs for comparison
    verbose=1
)

# Save the model
model.save('binaryCnn3-model.h5')

# Plot training and validation accuracy and loss
acc_aug = history_aug.history['accuracy']
val_acc_aug = history_aug.history['val_accuracy']
loss_aug = history_aug.history['loss']
val_loss_aug = history_aug.history['val_loss']

epochs_range_aug = range(len(acc_aug))

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range_aug, acc_aug, label='Training Accuracy (Aug)', marker='o')
plt.plot(epochs_range_aug, val_acc_aug, label='Validation Accuracy (Aug)', marker='x')
plt.legend(loc='lower right')
plt.title('Training vs Validation Accuracy (With Augmentation)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range_aug, loss_aug, label='Training Loss (Aug)', marker='o')
plt.plot(epochs_range_aug, val_loss_aug, label='Validation Loss (Aug)', marker='x')
plt.legend(loc='upper right')
plt.title('Training vs Validation Loss (With Augmentation)')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()

# Evaluate the model with augmented data
validation_generator.reset()
y_pred_aug = model.predict(validation_generator, verbose=1)
y_pred_class_aug = (y_pred_aug > 0.5).astype('int32').flatten()

# Compute confusion matrix and display results
y_true = validation_generator.classes[:len(y_pred_class_aug)]  # Match the lengths
cm_aug = confusion_matrix(y_true, y_pred_class_aug)

disp_labels = ['Healthy', 'Unhealthy']  # Updated labels
disp = ConfusionMatrixDisplay(confusion_matrix=cm_aug, display_labels=disp_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (With Augmentation)')
plt.show()

# Print classification report
print('Classification Report (With Augmentation)')
print(classification_report(y_true, y_pred_class_aug, target_names=disp_labels))

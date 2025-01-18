import os
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras.api import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import numpy as np


##### FIRST ITERATION #####
##### ACCURACY 68% #####

# Set the data directory
train_data_dir = './new-dataset'

# Create the data generator without data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    validation_split=0.1
)

# Training data generator
train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(64, 64),  # Use smaller images for simplicity
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True
)

# Validation data generator
validation_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Print the number of samples and batch sizes
print("Number of training samples:", train_generator.samples)
print("Number of validation samples:", validation_generator.samples)
print("Training batch size:", train_generator.batch_size)
print("Validation batch size:", validation_generator.batch_size)

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

# Train the model without specifying steps_per_epoch and validation_steps
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,  # Start with fewer epochs
    verbose=1
)

# Save the model
model.save('binaryCnn2-model.h5')

# Verify the lengths of the metrics
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

print("Length of training accuracy history:", len(acc))
print("Length of validation accuracy history:", len(val_acc))

# Adjust the plotting range
epochs_range = range(len(acc))

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy', marker='o')
plt.plot(epochs_range, val_acc, label='Validation Accuracy', marker='x')
plt.legend(loc='lower right')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss', marker='o')
plt.plot(epochs_range, val_loss, label='Validation Loss', marker='x')
plt.legend(loc='upper right')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()

# Evaluate the model on validation data
validation_generator.reset()
y_pred = model.predict(validation_generator, verbose=1)
y_pred_class = (y_pred > 0.5).astype('int32').flatten()

# Ensure y_pred_class and y_true lengths match
y_true = validation_generator.classes[:len(y_pred_class)]  # Trim y_true to match y_pred_class length if necessary

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred_class)

# Display confusion matrix
disp_labels = ['Healthy', 'Unhealthy']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print('Classification Report')
print(classification_report(y_true, y_pred_class, target_names=disp_labels))

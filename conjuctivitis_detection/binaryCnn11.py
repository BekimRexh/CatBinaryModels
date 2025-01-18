import os
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import numpy as np
from sklearn.utils import class_weight
from keras.api import regularizers
from keras.api.callbacks import ReduceLROnPlateau  # Import the callback

##### EIGHTH ITERATION #####
##### ACCURACY 92.31% again  #####
##### IMBALANCED ACCURACY 89% 5% WRONG UNHEALTHY 12% WRONG HEALHTY #####
##### NEW IMAGES OF CATS ADDED TO UNHEALTHY #####
##### IMBALANCED DATASET TEST TO FIX IMBALANCED ACCURACY #####
##### DROPOUT TO 0.5 #####
##### DEEPER MODEL #####
##### BATCH SIZE 4 #####
##### AUGMENTED DATASET #####
##### DROPOUT ADDED #####
##### ADDED DATA AUGMENTATION ######

##### VAL ACCURACY 85% ######

# Updated Data Generator (Normalization Only)
datagen = ImageDataGenerator(
    rescale=1./255,  # Only rescaling for normalization
    validation_split=0.15  # Keeping 10% for validation
)

# Training data generator
train_generator = datagen.flow_from_directory(
    './augmented-dataset-2',
    target_size=(256, 256),
    batch_size=4,  # Increase batch size for better optimization during fine-tuning
    class_mode='binary',
    subset='training',
    shuffle=True
)

# Validation data generator
validation_generator = datagen.flow_from_directory(
    './augmented-dataset-2',
    target_size=(256, 256),
    batch_size=4,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Verify the class indices
print("Class indices:", train_generator.class_indices)

# Step 3: Define the CNN Model
# Define a simple CNN model with dropout layers
# Enhanced model with additional convolutional layers
model = models.Sequential([
    layers.Conv2D(32, (3, 3), padding='same', input_shape=(256, 256, 3),
                  kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), padding='same',
                  kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), padding='same',
                  kernel_regularizer=regularizers.l2(0.001)),
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

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Calculate class weights to handle any remaining imbalance
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# Learning rate scheduler callback
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',  # Monitor validation loss
    factor=0.5,  # Reduce learning rate by a factor of 0.5
    patience=3,  # Number of epochs with no improvement before reducing the learning rate
    min_lr=1e-6,  # Lower bound on the learning rate
    verbose=1
)

# Step 4: Train the Model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=70,
    class_weight=class_weights,  # Apply class weights
    callbacks=[lr_scheduler],  # Add the learning rate scheduler
    verbose=1
)

# Save the model
model.save('binaryCnn11-model.h5')

# Step 5: Evaluate the Model and Plot Results
# Plot training and validation accuracy and loss
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
plt.title('Training vs Validation Accuracy (Balanced Dataset)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss', marker='o')
plt.plot(epochs_range, val_loss, label='Validation Loss', marker='x')
plt.legend(loc='upper right')
plt.title('Training vs Validation Loss (Balanced Dataset)')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()

# Step 6: Evaluate the Model with Validation Data
validation_generator.reset()
y_pred = model.predict(validation_generator, verbose=1)
y_pred_class = (y_pred > 0.5).astype('int32').flatten()

# Compute confusion matrix and display results
y_true = validation_generator.classes[:len(y_pred_class)]  # Match the lengths
cm = confusion_matrix(y_true, y_pred_class)

# Update labels based on class indices
class_indices = train_generator.class_indices
inv_class_indices = {v: k for k, v in class_indices.items()}
disp_labels = [inv_class_indices[0], inv_class_indices[1]]

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Balanced Dataset)')
plt.show()

# Print classification report
print('Classification Report (Balanced Dataset)')
print(classification_report(y_true, y_pred_class, target_names=disp_labels))

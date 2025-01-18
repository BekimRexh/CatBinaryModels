import os
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import numpy as np
from sklearn.utils import class_weight

##### FIFTH ITERATION #####
##### ACCURACY 86%  #####
##### NEW BALANCING DATASET BY AUGMENTING UNHEALTHY IMAGES #####
##### BATCH SIZE 8 #####
##### DROPOUT ADDED #####
##### ADDED DATA AUGMENTATION ######
##### INCREASED EPOCHS TO 15 ######


# Set the data directories
train_data_dir = './augmented-dataset'  # Main dataset directory (copy of revised-dataset)
unhealthy_data_dir = './augmented-dataset/2 Unhealthy'  # Unhealthy images directory

# Step 1: Data Augmentation for Unhealthy Images
# Create an ImageDataGenerator for augmentation specific to the unhealthy dataset
unhealthy_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load the unhealthy images for augmentation
unhealthy_generator = unhealthy_datagen.flow_from_directory(
    directory=os.path.dirname(unhealthy_data_dir),  # Parent directory of '2 Unhealthy'
    classes=['2 Unhealthy'],  # Focus on the '2 Unhealthy' class
    target_size=(64, 64),  # Use consistent target size
    batch_size=1,
    class_mode=None,
    shuffle=False,
    save_to_dir=unhealthy_data_dir,  # Save augmented images directly here
    save_prefix='aug',
    save_format='jpeg'
)

# Calculate the number of augmented images needed
num_unhealthy_images = len(os.listdir(unhealthy_data_dir))
num_healthy_images = len(os.listdir(os.path.join(train_data_dir, '1 Healthy')))
num_augmented_images_needed = num_healthy_images - num_unhealthy_images

print(f"Number of healthy images: {num_healthy_images}")
print(f"Number of unhealthy images: {num_unhealthy_images}")
print(f"Generating {num_augmented_images_needed} augmented images to balance the dataset...")

# Generate augmented images
for i in range(num_augmented_images_needed):
    unhealthy_generator.next()

print(f"Generated {num_augmented_images_needed} augmented images.")

# Step 2: Updated Data Generators for Training and Validation
# Create a new data generator for training and validation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1  # 10% of the data for validation
)

# Training data generator with both healthy and (augmented + original) unhealthy images
train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(64, 64),
    batch_size=8,
    class_mode='binary',
    subset='training',
    shuffle=True
)

# Validation data generator
validation_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(64, 64),
    batch_size=8,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Verify the class indices
print("Class indices:", train_generator.class_indices)

# Step 3: Define the CNN Model
# Define a simple CNN model with dropout layers
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
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

# Step 4: Train the Model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=15,
    class_weight=class_weights,  # Apply class weights
    verbose=1
)

# Save the model
model.save('binaryCnn6-model.h5')

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

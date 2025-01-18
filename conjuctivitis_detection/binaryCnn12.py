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
import seaborn as sns
from sklearn.metrics import classification_report
import pandas as pd

##### EIGHTH ITERATION #####
##### ACCURACY 92% #####
##### IMBALANCED ACCURACY 95.5% 5% WRONG UNHEALTHY 5% WRONG HEALHTY #####
##### NEW REDUCED TARGET SIZE #####
##### NEW REDUCED VALIDATION SPLIT ####
##### NEW INCREASED EPOCHS TO 70 #####
##### IMBALANCED TRAINING SET #####
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
    rescale=1./255,  
    validation_split=0.10  
)

# Training data generator
train_generator = datagen.flow_from_directory(
    './augmented-dataset-3',
    target_size=(64, 64),
    batch_size=4,  
    class_mode='binary',
    subset='training',
    shuffle=True
)

# Validation data generator
validation_generator = datagen.flow_from_directory(
    './augmented-dataset-3',
    target_size=(64, 64),
    batch_size=4,
    class_mode='binary',
    subset='validation',
    shuffle=False
)


print("Class indices:", train_generator.class_indices)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), padding='same', input_shape=(64, 64, 3),
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

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# Learning rate scheduler callback
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',  
    factor=0.5,  
    patience=3, 
    min_lr=1e-6,  
    verbose=1
)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=70,
    class_weight=class_weights,  
    callbacks=[lr_scheduler],  
    verbose=1
)

model.save('binaryCnn12-model.h5')

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
plt.title('Training vs Validation Loss (Balanced Dataset)')
plt.xlabel('Epochs')
plt.ylabel('Loss')

# Save the plot to your local directory
plt.savefig('training_validation_accuracy_loss.png', dpi=300)

plt.show()

# Evaluate the Model with Validation Data
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
plt.title('Confusion Matrix (Imbalanced Dataset)')

# Save the confusion matrix plot
plt.savefig('confusion_matrix.png', dpi=300)

plt.show()

# Generate classification report as a dictionary
report = classification_report(y_true, y_pred_class, target_names=disp_labels, output_dict=True)

# Convert the classification report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Exclude 'accuracy', 'macro avg', and 'weighted avg' if desired
report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'])

# Plot the classification report as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(report_df.iloc[:, :-1], annot=True, cmap='Blues', fmt='.2f', cbar=False)

plt.title('Classification Report (Imbalanced Dataset)')
plt.ylabel('Classes')
plt.xlabel('Metrics')

# Save the classification report plot
plt.savefig('classification_report.png', dpi=300, bbox_inches='tight')

plt.show()

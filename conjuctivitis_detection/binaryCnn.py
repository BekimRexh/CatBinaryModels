import os
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras.api.models import Model
from keras.api.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.api.applications import MobileNetV2
import numpy as np
import matplotlib.pyplot as plt


train_data_dir = './Revised dataset'

datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True, 
    fill_mode='nearest', 
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

def weighted_data_generator(generator):
    while True:
        images, labels = next(generator)
        sample_weights = []
        batch_size = len(images)
        for index in range(batch_size):
            filepath = generator.filepaths[generator.index_array[index]]
            if '1 Healthy' in filepath:
                sample_weights.append(1.0)
            elif '2 Low Chance' in filepath:
                sample_weights.append(1.25)
            elif '3 Moderate Chance' in filepath:
                sample_weights.append(1.75)
            elif '4 High Chance' in filepath:
                sample_weights.append(2.25)
            else:
                sample_weights.append(1.0)
        sample_weights = np.array(sample_weights)
        yield images, labels, sample_weights

base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_weighted_generator = weighted_data_generator(train_generator)

history = model.fit(
    train_weighted_generator,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    epochs=20,
    verbose=1
)

model.save('BINARY-cat_conjunctivitis_model.h5')

# **Results Plotting**: Plots the training and validation accuracy and loss after training is complete.
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(10, 5))

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
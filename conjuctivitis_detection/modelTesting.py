import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import os
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load the trained model
model = tf.keras.models.load_model('conjuctivitis_detection/finalConjunctModel.h5')

# Print model summary to verify input shape
# print(model.summary())

# Path to the test dataset
test_data_dir = 'conjuctivitis_detection/balanced-test-set'  # Update this with the correct path to your test dataset
# test_data_dir = 'conjuctivitis_detection/binary_test_dataset'  # Update this with the correct path to your test dataset
# test_data_dir = 'conjuctivitis_detection/binary_test_dataset_2'  # Update this with the correct path to your test dataset

# Create a folder to store misclassified images
misclassified_dir = 'conjuctivitis_detection/misclassified'
os.makedirs(misclassified_dir, exist_ok=True)

# Clear the misclassified folder if it already contains files
for filename in os.listdir(misclassified_dir):
    file_path = os.path.join(misclassified_dir, filename)
    if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)  # Remove the file
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)  # Remove directories

# Image data generator for the test set (no augmentation, only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load test data
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(512, 512),  # Updated to match the model's expected input size
    batch_size=1,  # Process one image at a time for easier analysis
    class_mode='binary',
    shuffle=False  # Don't shuffle, to keep file names and predictions aligned
)

# Get the file paths of the test images
file_paths = test_generator.filepaths

# Make predictions on the test data
predictions = model.predict(test_generator, verbose=1)
predicted_classes = (predictions > 0.5).astype(int).flatten()  # Convert probabilities to binary classes and flatten

# True labels
y_true = test_generator.classes  # True labels from the test generator

# Compute confusion matrix
cm = confusion_matrix(y_true, predicted_classes)

# Map numerical labels to class names
class_indices = test_generator.class_indices
inv_class_indices = {v: k for k, v in class_indices.items()}
class_names = [inv_class_indices[0], inv_class_indices[1]]

# Plot and save the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Test Set)')
plt.savefig('Confusion Matrix - Large Imbalanced Test Set.png', dpi=300)
plt.show()

# Generate classification report as a dictionary
report = classification_report(y_true, predicted_classes, target_names=class_names, output_dict=True)

# Convert the classification report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Exclude 'accuracy', 'macro avg', and 'weighted avg' if desired
report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'])

# Plot the classification report as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(report_df.iloc[:, :-1], annot=True, cmap='Blues', fmt='.2f', cbar=False)

plt.title('Classification Report (Test Set)')
plt.ylabel('Classes')
plt.xlabel('Metrics')

# Save the classification report plot
plt.savefig('Classification Report - Large Imbalanced Test Set.png', dpi=300, bbox_inches='tight')
plt.show()

# Variables to track correct predictions
correct_predictions = np.sum(predicted_classes == y_true)
total_images = len(predicted_classes)

# Map numerical labels to class names (if not already mapped)
class_labels = {0: 'healthy', 1: 'conjunctivitis'}

# Open a log file to store information about misclassified images
log_file_path = os.path.join(misclassified_dir, 'misclassified_log.txt')
with open(log_file_path, 'w') as log_file:
    log_file.write('Misclassified Images Log\n')
    log_file.write('Filename\tTrue Label\tPredicted Label\n')

    # Compare predictions with true labels and move misclassified images
    for i, (pred, label) in enumerate(zip(predicted_classes, y_true)):
        if pred != label:
            # Get the original file path of the misclassified image
            original_file_path = file_paths[i]
            # Create the new file path in the misclassified folder
            new_file_path = os.path.join(misclassified_dir, os.path.basename(original_file_path))
            # Copy the misclassified image to the new folder
            shutil.copy(original_file_path, new_file_path)
            # Log the misclassification details with class names
            log_file.write(f"{os.path.basename(original_file_path)}\t{class_labels[label]}\t{class_labels[pred]}\n")

# Calculate test accuracy
test_accuracy = (correct_predictions / total_images) * 100

# Print the test accuracy
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Misclassified images have been copied to: {misclassified_dir}")

# Save classification report to a text file (optional)
report_text = classification_report(y_true, predicted_classes, target_names=class_names)
with open('classification_report_test.txt', 'w') as f:
    f.write('Classification Report (Large Imbalanced Test Set)\n\n')
    f.write(report_text)

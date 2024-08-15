import os
import numpy as np
from PIL import Image

# Load and preprocess training, validation, and testing data
training_dir = '/home/anjalimedini/Downloads/training/'
validation_dir = '/home/anjalimedini/Downloads/validation/'
testing_dir = '/home/anjalimedini/Downloads/testing/'

training_data = []
training_labels = []
validation_data = []
validation_labels = []
testing_data = []
testing_labels = []

for filename in os.listdir(training_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img_path = os.path.join(training_dir, filename)
        img_array = np.array(Image.open(img_path).convert('L').resize((28, 28))).flatten()
        training_data.append(img_array)
        training_labels.append(1)  # Example label, adjust based on your dataset

for filename in os.listdir(validation_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img_path = os.path.join(validation_dir, filename)
        img_array = np.array(Image.open(img_path).convert('L').resize((28, 28))).flatten()
        validation_data.append(img_array)
        validation_labels.append(1)  # Example label, adjust based on your dataset

for filename in os.listdir(testing_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img_path = os.path.join(testing_dir, filename)
        img_array = np.array(Image.open(img_path).convert('L').resize((28, 28))).flatten()
        testing_data.append(img_array)
        testing_labels.append(1)  # Example label, adjust based on your dataset

# Hyperparameters
epochs = 5
learning_rate = 0.01
momentum = 0.9
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Train using SGD
weights_sgd = np.random.randn(len(training_data[0]))
bias_sgd = 0.1
training_accuracy_sgd = []
for epoch in range(epochs):
    correct = 0
    for i in range(len(training_data)):
        prediction = np.dot(weights_sgd, training_data[i]) + bias_sgd
        prediction = 1 if prediction >= 0 else 0
        error = training_labels[i] - prediction
        weights_sgd += learning_rate * error * training_data[i]
        bias_sgd += learning_rate * error
        correct += 1 if prediction == training_labels[i] else 0
    training_accuracy_sgd.append(correct / len(training_data))
    print(f"SGD Epoch {epoch+1}, Training Accuracy: {training_accuracy_sgd[-1]:.2f}")

# Evaluate SGD on validation data
validation_predictions_sgd = np.dot(validation_data, weights_sgd) + bias_sgd
validation_predictions_sgd = (validation_predictions_sgd >= 0).astype(int)
validation_accuracy_sgd = np.mean(validation_predictions_sgd == validation_labels)
print(f"SGD Validation Accuracy: {validation_accuracy_sgd:.2f}")

# Evaluate SGD on testing data
testing_predictions_sgd = np.dot(testing_data, weights_sgd) + bias_sgd
testing_predictions_sgd = (testing_predictions_sgd >= 0).astype(int)
testing_accuracy_sgd = np.mean(testing_predictions_sgd == testing_labels)
print(f"SGD Testing Accuracy: {testing_accuracy_sgd:.2f}")

# Train using SGDM
weights_sgdm = np.random.randn(len(training_data[0]))
bias_sgdm = 0
velocity = np.zeros(len(training_data[0]))
training_accuracy_sgdm = []
for epoch in range(epochs):
    correct = 0
    for i in range(len(training_data)):
        prediction = np.dot(weights_sgdm, training_data[i]) + bias_sgdm
        prediction = 1 if prediction >= 0 else 0
        error = training_labels[i] - prediction
        velocity = momentum * velocity + learning_rate * error * training_data[i]
        weights_sgdm += velocity
        bias_sgdm += learning_rate * error
        correct += 1 if prediction == training_labels[i] else 0
    training_accuracy_sgdm.append(correct / len(training_data))
    print(f"SGDM Epoch {epoch+1}, Training Accuracy: {training_accuracy_sgdm[-1]:.2f}")

# Evaluate SGD on validation data
validation_predictions_sgdm = np.dot(validation_data, weights_sgd) + bias_sgd
validation_predictions_sgdm = (validation_predictions_sgdm >= 0).astype(int)
validation_accuracy_sgdm = np.mean(validation_predictions_sgdm == validation_labels)
print(f"SGDM Validation Accuracy: {validation_accuracy_sgd:.2f}")

# Evaluate SGD on testing data
testing_predictions_sgdm = np.dot(testing_data, weights_sgd)

testing_predictions_sgdm = np.dot(testing_data, weights_sgd) + bias_sgd
testing_predictions_sgdm = (testing_predictions_sgdm >= 0).astype(int)
testing_accuracy_sgdm = np.mean(testing_predictions_sgdm == testing_labels)
print(f"SGDM Testing Accuracy: {testing_accuracy_sgd:.2f}")

# Train using Adam
weights_adam = np.random.randn(len(training_data[0]))
bias_adam = 0.1
m = np.zeros(len(training_data[0]))
v = np.zeros(len(training_data[0]))
training_accuracy_adam = []
for epoch in range(epochs):
    correct = 0
    for i in range(len(training_data)):
        prediction = np.dot(weights_adam, training_data[i]) + bias_adam
        prediction = 1 if prediction >= 0 else 0
        error = training_labels[i] - prediction
        m = beta1 * m + (1 - beta1) * error * training_data[i]
        v = beta2 * v + (1 - beta2) * error ** 2 * training_data[i]
        weights_adam += learning_rate * m / (np.sqrt(v) + epsilon)
        bias_adam += learning_rate * error
        correct += 1 if prediction == training_labels[i] else 0
    training_accuracy_adam.append(correct / len(training_data))
    print(f"Adam Epoch {epoch+1}, Training Accuracy: {training_accuracy_adam[-1]:.2f}")

# Evaluate Adam on validation data
validation_predictions_adam = np.dot(validation_data, weights_adam) + bias_adam
validation_predictions_adam = (validation_predictions_adam >= 0).astype(int)
validation_accuracy_adam = np.mean(validation_predictions_adam == validation_labels)
print(f"Adam Validation Accuracy: {validation_accuracy_adam:.2f}")

# Evaluate Adam on testing data
testing_predictions_adam = np.dot(testing_data, weights_adam) + bias_adam
testing_predictions_adam = (testing_predictions_adam >= 0).astype(int)
testing_accuracy_adam = np.mean(testing_predictions_adam == testing_labels)
print(f"Adam Testing Accuracy: {testing_accuracy_adam:.2f}")
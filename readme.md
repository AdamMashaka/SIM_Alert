# Data Security Alert System

Welcome to the Data Security Alert System repository! This project aims to provide a robust solution for identifying data protection leakage using machine learning techniques. The system includes user registration with fingerprint data and alerts when someone accesses data in the database.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methods](#methods)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Data Security Alert System is designed to help users register their SIM card details securely, including capturing and storing fingerprint data. Additionally, the system includes a machine learning model for recognizing unauthorized access to data. This project leverages advanced machine learning techniques to ensure data security and accurate detection of unauthorized access.

## Features

- **User Registration**: Register users with their SIM card details and fingerprint data.
- **Fingerprint Capture**: Capture and store fingerprint images securely.
- **Unauthorized Access Detection**: Detect and alert when someone accesses data in the database.
- **Database Management**: Store and manage user data securely using SQLite.

## Installation

To get started with the Data Security Alert System, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/data-security-alert-system.git
    cd data-security-alert-system
    ```

2. **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the application**:
    ```bash
    streamlit run main.py
    ```

## Usage

### Home Page

The home page provides an overview of the system and its features. It includes a brief introduction and instructions on how to use the system.

### Register

The registration page allows users to register their SIM card details and capture their fingerprint data. Users need to fill in their full name, location, NIDA number, phone number, and password. They can also upload a fingerprint image.

### Database

The database page allows administrators to monitor access to the data. Alerts are generated when unauthorized access is detected.

### About

The about page provides information about the project, the team, and the mission. It also includes contact information for feedback and inquiries.

## Methods

### `model_prediction(test_image)`

This method loads a pre-trained TensorFlow model and uses it to predict unauthorized access based on the uploaded fingerprint image. It returns the index of the predicted class.

### `User` Class

The `User` class defines the user model for registration. It includes fields for storing user details and fingerprint data. The class also includes methods for setting and checking passwords.

### Fingerprint Capture

The fingerprint capture section allows users to upload a fingerprint image. The image is processed and stored in the database as a binary blob.

### Registration Logic

The registration logic checks if all fields are filled and if the NIDA number already exists in the database. If the NIDA number is unique, the user details and fingerprint data are stored in the database.

## Model Training

To train the machine learning model for unauthorized access detection, follow these steps:

1. **Preprocess Fingerprint Images**:
    - Load images from the specified directory.
    - Convert images to grayscale, resize them to 128x128, and normalize pixel values.
    - Extract labels from filenames.

2. **Build and Train the CNN Model**:
    - Define a CNN model using Keras.
    - Ensure the number of classes in the final dense layer matches the number of unique labels.
    - Compile the model with the Adam optimizer and categorical crossentropy loss.
    - Train the model on the preprocessed images.

3. **Save and Evaluate the Model**:
    - Save the trained model to a file named `fingerprint_recognition_model.h5`.
    - Evaluate the model's performance on the test set.

4. **Optional: Load and Use the Model**:
    - Load the saved model and use it for predictions.

## Contributing

We welcome contributions to the Data Security Alert System! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Improve the Formulae of Accuracy

To improve the accuracy of the fingerprint recognition model, consider the following strategies:

### 1. Data Augmentation
Data augmentation can help improve the robustness and generalization of your model by artificially increasing the size and diversity of your training dataset. Apply transformations such as rotation, scaling, translation, and flipping to your fingerprint images.

### 2. Hyperparameter Tuning
Experiment with different hyperparameters such as learning rate, batch size, number of epochs, and optimizer. Use techniques like grid search or random search to find the optimal hyperparameters for your model.

### 3. Model Architecture
Consider experimenting with different neural network architectures. Try deeper networks, different types of layers (e.g., convolutional layers, dropout layers), and different activation functions.

### 4. Regularization
Regularization techniques such as dropout, L1/L2 regularization, and batch normalization can help prevent overfitting and improve the generalization of your model.

### 5. Transfer Learning
Leverage pre-trained models on similar tasks and fine-tune them on your fingerprint dataset. Transfer learning can significantly improve accuracy, especially when you have a limited amount of training data.

### 6. Cross-Validation
Use cross-validation to evaluate your model's performance more reliably. This can help you detect overfitting and ensure that your model generalizes well to unseen data.

### 7. Ensemble Methods
Combine predictions from multiple models to improve accuracy. Ensemble methods such as bagging, boosting, and stacking can help reduce variance and bias.

### Example: Data Augmentation and Hyperparameter Tuning

Here is an example of how you can implement data augmentation and hyperparameter tuning in your training script:

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load your dataset
# X, y = load_your_dataset()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # Adjust the number of classes as needed
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with data augmentation
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=50, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')

# Save the model
model.save('fingerprint_recognition_model.h5')

```



### Add more metrices for classification

1. **Additional Metrics**:
   - Added `tf.keras.metrics.Precision()`, `tf.keras.metrics.Recall()`, and `tf.keras.metrics.AUC()` to the [`metrics`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fadam%2FDownloads%2FSIM_Alert%2Freadme.md%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A187%2C%22character%22%3A91%7D%7D%5D%2C%22b3a138b4-5fb9-4bc5-8ce8-84ea1b4f11c0%22%5D "Go to definition") parameter in the `model.compile()` method.

2. **Evaluation**:
   - Evaluated the model on the test set and printed the accuracy, precision, recall, and AUC.
   - Generated a classification report and confusion matrix using `classification_report` and `confusion_matrix` from `sklearn.metrics`.

### Data Collection methodology
 
A robust data collection methodology is crucial for training an accurate and reliable fingerprint recognition model. Here are the steps to collect, preprocess, and manage the data:

1. **Data Collection**:
    - Collect fingerprint images from a diverse set of individuals to ensure variability in the dataset.
    - Ensure that the images are captured under different conditions (e.g., lighting, angle) to improve the model's robustness.

2. **Data Annotation**:
    - Label each fingerprint image with a unique identifier corresponding to the individual.
    - Use consistent and clear labeling conventions to avoid confusion during training.

3. **Data Preprocessing**:
    - Convert the images to grayscale to reduce complexity and focus on the fingerprint patterns.
    - Resize the images to a consistent size (e.g., 128x128 pixels) to ensure uniformity in the input data.
    - Normalize the pixel values to a range of [0, 1] to improve the model's convergence during training.

4. **Data Augmentation**:
    - Apply data augmentation techniques to artificially increase the size and diversity of the training dataset.
    - Use transformations such as rotation, scaling, translation, and flipping to create variations of the original images.

5. **Data Splitting**:
    - Split the dataset into training, validation, and testing sets to evaluate the model's performance.
    - Use a common split ratio (e.g., 70% training, 20% validation, 10% testing) to ensure a balanced evaluation.

6. **Data Storage**:
    - Organize the data into separate directories for training, validation, and testing sets.
    - Use a consistent directory structure to facilitate easy loading and management of the data.


# 
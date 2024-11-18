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


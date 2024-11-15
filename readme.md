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


# table generation 
The error `sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) no such column: user.location` indicates that the [`location`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fadam%2FDownloads%2FSIM_Alert%2Fmain.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A139%2C%22character%22%3A42%7D%7D%5D%2C%2229de086e-7807-4224-8505-34a5005894fd%22%5D "Go to definition") column does not exist in the [`user`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fadam%2FDownloads%2FSIM_Alert%2Fmain.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A143%2C%22character%22%3A28%7D%7D%5D%2C%2229de086e-7807-4224-8505-34a5005894fd%22%5D "Go to definition") table. This can happen if the database schema was not updated after adding the [`location`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fadam%2FDownloads%2FSIM_Alert%2Fmain.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A139%2C%22character%22%3A42%7D%7D%5D%2C%2229de086e-7807-4224-8505-34a5005894fd%22%5D "Go to definition") column to the [`User`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fadam%2FDownloads%2FSIM_Alert%2Fmain.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A139%2C%22character%22%3A26%7D%7D%5D%2C%2229de086e-7807-4224-8505-34a5005894fd%22%5D "Go to definition") model.

To resolve this issue, you need to ensure that the database schema is updated to include the [`location`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fadam%2FDownloads%2FSIM_Alert%2Fmain.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A139%2C%22character%22%3A42%7D%7D%5D%2C%2229de086e-7807-4224-8505-34a5005894fd%22%5D "Go to definition") column. You can do this by dropping the existing table and creating a new one, or by using a migration tool like Alembic to handle schema changes.

For simplicity, let's drop the existing table and create a new one with the updated schema. Note that this will delete all existing data in the table.

### Updated file:///home/adam/Downloads/SIM_Alert/main.py

```python
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, LargeBinary
from sqlalchemy.orm import declarative_base, sessionmaker
import tensorflow as tf
import numpy as np
import cv2
import subprocess
import pandas as pd

# Database setup for SQLite
DATABASE_URL = "sqlite:///data_alert.db"  # SQLite database file
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session = SessionLocal()
Base = declarative_base()

# Define User model for user registration
class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    location = Column(String, nullable=False)
    nida_number = Column(String, unique=True, nullable=False)
    phone_number = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    fingerprint = Column(LargeBinary, nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Drop the existing table and create a new one with the updated schema
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("fingerprint_recognition_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Function to capture fingerprint using fprintd
def capture_fingerprint():
    try:
        # Enroll fingerprint using fprintd
        subprocess.run(["fprintd-enroll"], check=True)
        # Verify fingerprint and capture the image
        result = subprocess.run(["fprintd-verify"], capture_output=True, text=True, check=True)
        if "verify-match" in result.stdout:
            # Capture the fingerprint image
            fingerprint_image = subprocess.run(["fprintd-list"], capture_output=True, text=True, check=True)
            return fingerprint_image.stdout.encode()
        else:
            st.error("Fingerprint verification failed.")
            return None
    except subprocess.CalledProcessError as e:
        st.error(f"Error capturing fingerprint: {e}")
        return None

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Register", "Database", "About"])

# Main Page
if app_mode == "Home":  # Home Page
    st.header("Data Security Alert System")
    image_path = "jisajili.jpeg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Data Security Alert System! 🎒📳
    
    Our mission is to solve in identifying data protection leakage. Register your SIM card details and capture your fingerprint to ensure secure access to your data.

    ### How It Works
    1. **Register:** Go to the **Register** page and fill in your details.
    2. **Capture Fingerprint:** Upload your fingerprint image or use your laptop's fingerprint scanner for secure registration.
    3. **Monitor Access:** Administrators can monitor access to the data and receive alerts for unauthorized access.

    ### Why Choose Us?
    - **Security:** Advanced machine learning techniques for secure data access.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Quick registration and monitoring process.
    """)

elif app_mode == "Register":
    st.header("Register your SIM card details here")

    # Step 2: User Registration Form
    st.title("User Registration")

    name = st.text_input("Full Name")
    location = st.selectbox("Location", ["Dar es Salaam", "Morogoro", "Mwanza", "Arusha"])
    nida_number = st.text_input("NIDA Number")
    phone_number = st.text_input("Phone Number")
    password = st.text_input("Password", type="password")

    # Fingerprint capture
    st.subheader("Capture Fingerprint")
    fingerprint_image = st.file_uploader("Upload Fingerprint Image", type=["png", "jpg", "bmp"])
    use_scanner = st.checkbox("Use Laptop Fingerprint Scanner")

    # Check if registration button is clicked
    if st.button("Register"):
        if name and location and nida_number and phone_number and password and (fingerprint_image or use_scanner):
            # Check if the NIDA number already exists
            user = session.query(User).filter_by(nida_number=nida_number).first()
            if user:
                st.error("A user with that NIDA number already exists.")
            else:
                if use_scanner:
                    fingerprint_data = capture_fingerprint()
                else:
                    # Read the fingerprint image
                    file_bytes = np.asarray(bytearray(fingerprint_image.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                    _, buffer = cv2.imencode('.bmp', img)
                    fingerprint_data = buffer.tobytes()

                if fingerprint_data:
                    # Create a new user and add to the database
                    new_user = User(name=name, location=location, nida_number=nida_number, phone_number=phone_number, fingerprint=fingerprint_data)
                    new_user.set_password(password)  # Hash the password
                    session.add(new_user)
                    session.commit()
                    st.success("Registration successful!")
        else:
            st.error("Please fill in all fields.")

elif app_mode == "Database":
    st.header("Monitor Access by Location")
    location = st.selectbox("Select Location", ["Dar es Salaam", "Morogoro", "Mwanza", "Arusha"])

    # Query users by location
    users = session.query(User).filter_by(location=location).all()

    if users:
        # Display users in a table
        df = pd.DataFrame([(user.name, user.location, user.nida_number, user.phone_number) for user in users],
                          columns=["Name", "Location", "NIDA Number", "Phone Number"])
        st.dataframe(df)

        # Download data as CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download data as CSV", data=csv, file_name=f'{location}_users.csv', mime='text/csv')

        # Print data
        if st.button("Print Data"):
            st.write(df.to_html(), unsafe_allow_html=True)
    else:
        st.write("No users found for this location.")

elif app_mode == "About":
    st.header("About Us")
    st.markdown("""
    Learn more about the project, team, and our goals.

    ### About Us
    We are a team of dedicated individuals working towards improving data security through advanced technology. Our Data Security Alert System is designed to help users securely register their SIM card details and monitor access to their data.

    ### Our Mission
    Our mission is to leverage the power of machine learning and artificial intelligence to provide accurate and efficient data security. We aim to make this technology accessible to everyone, ensuring secure data access for all.

    ### Contact Us
    If you have any questions or feedback, feel free to reach out to us at [contact@example.com](mailto:contact@example.com).
    """)
```

### Explanation

1. **Drop and Create Table**:
   - Dropped the existing [`user`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fadam%2FDownloads%2FSIM_Alert%2Fmain.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A143%2C%22character%22%3A28%7D%7D%5D%2C%2229de086e-7807-4224-8505-34a5005894fd%22%5D "Go to definition") table and created a new one with the updated schema to ensure the [`location`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fadam%2FDownloads%2FSIM_Alert%2Fmain.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A139%2C%22character%22%3A42%7D%7D%5D%2C%2229de086e-7807-4224-8505-34a5005894fd%22%5D "Go to definition") column is included.

2. **User Registration Form**:
   - Added a dropdown to select the user's location from ["Dar es Salaam", "Morogoro", "Mwanza", "Arusha"].

3. **Database Section**:
   - Added a dropdown to select the location for monitoring access.
   - Queried users by the selected location and displayed the results in a table.
   - Added options to download the data as a CSV file and print the data.

4. **Main Page**:
   - Updated the content to reflect the correct functionality of the system, focusing on SIM card registration and data access monitoring by location.

### Note

- The `fprintd` service is used for fingerprint scanning on Linux. Ensure that `fprintd` is installed and configured on your system.
- This example assumes that the fingerprint scanner is supported by `fprintd`. If you are using a different operating system or fingerprint scanner, you will need to use the appropriate library or SDK for that hardware.

Make sure to replace `'extraction/train_data'` with the actual path to your fingerprint images. Run this script in your Jupyter notebook to preprocess the data, train the model, and save the trained model. The additional checks will help ensure that the images and labels are loaded correctly.
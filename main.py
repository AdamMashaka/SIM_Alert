from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, LargeBinary
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
import tensorflow as tf
import numpy as np
import cv2
import subprocess
import pandas as pd

# Database setup for SQLite
DATABASE_URL = "sqlite:///data_alert.db"  # SQLite database file
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
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
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

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
            session = SessionLocal()
            try:
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
                        st.write(f"Registered user: {new_user.name}, Location: {new_user.location}")
                    else:
                        st.error("Failed to capture fingerprint.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                session.close()
        else:
            st.error("Please fill in all fields.")

elif app_mode == "Database":
    st.header("Monitor Access by Location")
    location = st.selectbox("Select Location", ["Dar es Salaam", "Morogoro", "Mwanza", "Arusha"])

    session = SessionLocal()
    try:
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
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        session.close()

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

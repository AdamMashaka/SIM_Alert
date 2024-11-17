from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, LargeBinary
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import io
import base64
import africastalking
import tensorflow as tf

# Database setup for SQLite
DATABASE_URL = "sqlite:///data_alert.db"  # SQLite database file
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()

# Africa's Talking setup
AT_USERNAME = "simalert"
AT_API_KEY = "atsk_3d33d5c86633640ad2f2c417e22ed471a7584943ce3ba8ad9b78a3d035f984f6ab9d596d"
africastalking.initialize(AT_USERNAME, AT_API_KEY)
sms = africastalking.SMS

# Load TensorFlow model
model = tf.keras.models.load_model("fingerprint_recognition_model.h5")

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

# Ensure tables exist
Base.metadata.create_all(bind=engine)

# Function to send SMS alerts using Africa's Talking
def send_sms_alert(phone_number, location):
    # Ensure phone number is in E.164 format
    if not phone_number.startswith('+'):
        phone_number = '+255' + phone_number.lstrip('0')

    message = f"Alert: Your data in {location} has been accessed."
    try:
        # Send SMS
        response = sms.send(message, [phone_number])
        st.success(f"SMS sent to {phone_number}: {response}")
        return response
    except Exception as e:
        st.error(f"Failed to send SMS to {phone_number}: {e}")

# Function to predict fingerprint
def predict_fingerprint(image):
    image = cv2.resize(image, (128, 128))
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return np.argmax(prediction)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Register", "Database", "About"])

# Main Page
if app_mode == "Home":  # Home Page
    st.header("Data Security Alert System")
    image_path = "jisajili.jpeg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    <div style="text-align: center;">
        <h1>Welcome to the Data Security Alert System! 🎒📳</h1>
        <p>Our mission is to identify and prevent data protection leakage. Register your SIM card details and capture your fingerprint to ensure secure access to your data.</p>
    </div>
    """, unsafe_allow_html=True)

elif app_mode == "Register":
    st.header("Register your SIM card details here")

    # User Registration Form
    st.title("User Registration")

    name = st.text_input("Full Name")
    location = st.selectbox("Location", ["Dar es Salaam", "Morogoro", "Mwanza", "Arusha"])
    nida_number = st.text_input("NIDA Number")
    phone_number = st.text_input("Phone Number")
    password = st.text_input("Password", type="password")

    # Fingerprint capture
    st.subheader("Capture Fingerprint")
    fingerprint_image = st.file_uploader("Upload Fingerprint Image", type=["png", "jpg", "bmp"])

    # Check if registration button is clicked
    if st.button("Register"):
        if name and location and nida_number and phone_number and password and fingerprint_image:
            session = SessionLocal()
            try:
                # Check if the NIDA number already exists
                user = session.query(User).filter_by(nida_number=nida_number).first()
                if user:
                    st.error("A user with that NIDA number already exists.")
                else:
                    # Read the fingerprint image
                    file_bytes = np.asarray(bytearray(fingerprint_image.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                    _, buffer = cv2.imencode('.bmp', img)
                    fingerprint_data = buffer.tobytes()

                    # Predict fingerprint
                    prediction = predict_fingerprint(img)
                    st.write(f"Fingerprint prediction: {prediction}")

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
            data = []
            for user in users:
                img_html = "No Image"
                if user.fingerprint:
                    fingerprint_image = Image.open(io.BytesIO(user.fingerprint))
                    buffered = io.BytesIO()
                    fingerprint_image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    img_html = f'<img src="data:image/png;base64,{img_str}" width="100" height="100">'

                row = {
                    "Name": user.name,
                    "Location": user.location,
                    "NIDA Number": user.nida_number,
                    "Phone Number": user.phone_number,
                    "Fingerprint": img_html
                }
                data.append(row)

            df = pd.DataFrame(data)

            # Convert the DataFrame to HTML
            df_html = df.to_html(escape=False)

            # Display the DataFrame as HTML
            st.markdown(df_html, unsafe_allow_html=True)

            # Download data as CSV and send SMS alerts
            if st.download_button(label="Download data as CSV", data=df.drop(columns=["Fingerprint"]).to_csv(index=False).encode('utf-8'), file_name=f'{location}_users.csv', mime='text/csv'):
                for user in users:
                    send_sms_alert(user.phone_number, location)
        else:
            st.write(f"No users found for this location: {location}.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        session.close()

elif app_mode == "About":
    st.header("About Us")
    st.markdown("""
    ## Welcome to the Data Security Alert System!

    At Data Security Alert System, we are committed to enhancing data security through cutting-edge technology. Our platform is designed to help users securely register their SIM card details and monitor access to their data, ensuring peace of mind and protection against unauthorized access.

    ### Our Vision
    To be the leading provider of innovative data security solutions, empowering individuals and organizations to safeguard their sensitive information with confidence.

    ### Our Mission
    Our mission is to leverage the power of machine learning and artificial intelligence to deliver accurate and efficient data security solutions. We strive to make this technology accessible to everyone, ensuring secure data access for all.

    ### Our Values
    - **Innovation:** Continuously pushing the boundaries of technology to provide the best security solutions.
    - **Integrity:** Upholding the highest standards of honesty and transparency in all our actions.
    - **Customer Focus:** Putting our users at the heart of everything we do, ensuring their needs are met with excellence.
    - **Collaboration:** Working together as a team and with our partners to achieve common goals.

    ### Meet the Team
    Our team consists of dedicated professionals with expertise in data security, machine learning, and software development. We are passionate about creating solutions that make a difference.

    ### Contact Us
    We would love to hear from you! If you have any questions, feedback, or inquiries, please feel free to reach out to us.

    - **Email:** [contact@example.com](mailto:contact@example.com)
    - **Phone:** +123-456-7890
    - **Address:** 123 Data Security Lane, Tech City, TX 75001

    ### Follow Us
    Stay connected and follow us on social media for the latest updates and news:
    - [LinkedIn](https://www.linkedin.com)
    - [Twitter](https://www.twitter.com)
    - [Facebook](https://www.facebook.com)

    Thank you for choosing Data Security Alert System. Together, let's build a safer digital world!
    """)
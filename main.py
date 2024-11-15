from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, LargeBinary
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
import numpy as np
import cv2
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

# Ensure tables exist
Base.metadata.create_all(bind=engine)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Register", "Database", "About"])

# Debugging: Print current database content
def debug_print_all_users():
    session = SessionLocal()
    users = session.query(User).all()
    for user in users:
        print(f"User: {user.name}, Location: {user.location}, NIDA: {user.nida_number}")
    session.close()

if app_mode == "Home":  # Home Page
    st.header("Data Security Alert System")
    image_path = "jisajili.jpeg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Data Security Alert System! 🎒📳
    
    Our mission is to solve in identifying data protection leakage. Register your SIM card details and capture your fingerprint to ensure secure access to your data.

    ### How It Works
    1. **Register:** Go to the **Register** page and fill in your details.
    2. **Capture Fingerprint:** Upload your fingerprint image for secure registration.
    3. **Monitor Access:** Administrators can monitor access to the data and receive alerts for unauthorized access.

    ### Why Choose Us?
    - **Security:** Advanced machine learning techniques for secure data access.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Quick registration and monitoring process.
    """)

elif app_mode == "Register":
    st.header("Register your SIM card details here")

    name = st.text_input("Full Name")
    location = st.selectbox("Location", ["Dar es Salaam", "Morogoro", "Mwanza", "Arusha"])
    nida_number = st.text_input("NIDA Number")
    phone_number = st.text_input("Phone Number")
    password = st.text_input("Password", type="password")
    fingerprint_image = st.file_uploader("Upload Fingerprint Image", type=["png", "jpg", "bmp"])

    if st.button("Register"):
        if name and location and nida_number and phone_number and password and fingerprint_image:
            session = SessionLocal()
            try:
                # Check if user already exists
                user = session.query(User).filter_by(nida_number=nida_number).first()
                if user:
                    st.error("A user with that NIDA number already exists.")
                else:
                    # Process fingerprint data
                    file_bytes = np.asarray(bytearray(fingerprint_image.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                    _, buffer = cv2.imencode('.bmp', img)
                    fingerprint_data = buffer.tobytes()

                    # Create new user
                    new_user = User(
                        name=name.strip(),
                        location=location.strip().title(),
                        nida_number=nida_number.strip(),
                        phone_number=phone_number.strip(),
                        fingerprint=fingerprint_data,
                    )
                    new_user.set_password(password.strip())
                    session.add(new_user)
                    session.commit()

                    st.success(f"User '{new_user.name}' registered successfully.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                session.close()

            # Debugging: Print all users
            debug_print_all_users()
        else:
            st.error("Please fill in all required fields.")

elif app_mode == "Database":
    st.header("Monitor Access by Location")
    location = st.selectbox("Select Location", ["Dar es Salaam", "Morogoro", "Mwanza", "Arusha"])
    
    session = SessionLocal()
    try:
        # Standardize location formatting
        location = location.strip().title()
        
        # Query users by location
        users = session.query(User).filter_by(location=location).all()

        if users:
            df = pd.DataFrame(
                [(user.name, user.location, user.nida_number, user.phone_number) for user in users],
                columns=["Name", "Location", "NIDA Number", "Phone Number"]
            )
            st.dataframe(df)
        else:
            st.write(f"No users found for this location: {location}.")
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

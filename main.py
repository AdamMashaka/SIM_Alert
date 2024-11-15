from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker 
import tensorflow as tf
import numpy as np 




# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Register", "Database", "Contacts"])

# Main Page
if app_mode == "Home": # Home Page
    st.header("MACHINE LEARNING(ML) OUR ALERT MODEL")
    image_path = "jisajili.jpeg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to Data security alert system! 🎒📳
    
    Our mission is to solve in identifying data protection leakage. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.
    """)

elif app_mode == "Register":
    st.header("Register your Sim card details here")
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

        def set_password(self, password):
            self.password_hash = generate_password_hash(password)

        def check_password(self, password):
            return check_password_hash(self.password_hash, password)

    # Create tables in the database (if they don’t already exist)
    Base.metadata.create_all(bind=engine)

    # Step 2: User Registration Form
    st.title("User Registration")

    name = st.text_input("Full Name")
    location = st.text_input("Location")
    nida_number = st.text_input("NIDA Number")
    phone_number = st.text_input("Phone Number")
    password = st.text_input("Password", type="strong password")

    # Check if registration button is clicked
    if st.button("Register"):
        if name and location and nida_number and phone_number and password:
            # Check if the NIDA number already exists
            user = session.query(User).filter_by(nida_number=nida_number).first()
            if user:
                st.error("A user with that NIDA number already exists.")
            else:
                # Create a new user and add to the database
                new_user = User(name=name, location=location, nida_number=nida_number, phone_number=phone_number)
                new_user.set_password(password)  # Hash the password
                session.add(new_user)
                session.commit()
                st.success("Registration successful!")
        else:
            st.error("Please fill in all fields.")

elif app_mode == "Database":
    st.header("Alert Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        st.image(test_image, use_container_width=True)
    # Predict button
    if st.button("Predict"):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        # Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                      'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                      'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                      'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                      'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                      'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                      'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                      'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                      'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                      'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                      'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                      'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                      'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Model is Predicting it's not {}".format(class_name[result_index]))

elif app_mode == "About":
    st.header("About Us")
    st.markdown("""
    Learn more about the project, team, and our goals.

    ### About Us
    We are a team of dedicated individuals working towards improving plant health through advanced technology. Our Plant Disease Recognition System is designed to help farmers and gardeners quickly identify and address plant diseases, ensuring healthier crops and better yields.

    ### Our Mission
    Our mission is to leverage the power of machine learning and artificial intelligence to provide accurate and efficient plant disease detection. We aim to make this technology accessible to everyone, from small-scale gardeners to large agricultural enterprises.

    ### Contact Us
    If you have any questions or feedback, feel free to reach out to us at [contact@example.com](mailto:contact@example.com).
    """)
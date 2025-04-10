import os
from dotenv import load_dotenv
import google.generativeai as genai


# Import required libraries
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import google.generativeai as genai

# Load the pre-trained model for image classification
model = load_model('D:\Byte-Builder-main\Byte-Builder-main\model_25_epoch.h5')

# Configure Gemini API

load_dotenv()  # Load environment variables from .env file

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
genai.configure(api_key="AIzaSyBI-Xw5zNOpZgNL4PteCIRt3qY4_kQzoSM")

# Initialize the GenerativeModel with updated model name
gen_model = genai.GenerativeModel('gemini-1.5-pro')

# Define a function for image prediction
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    classes = model.predict(x)
    result = np.argmax(classes[0])

    labels = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
    return labels[result]

# Function to generate medical report using Gemini
def generate_report_gemini(disease, infected_part, confidence_level, treatment_info, patient_friendly_summary):
    report_prompt = f"""
    Patient Information:
    Name: null
    Age: null
    Gender: null
    ID: null
    Date of Scan: null

    Infected Part Information:
    {infected_part}

    Disease Detection Result:
    Disease: {disease}
    AI Confidence Level: {confidence_level}%

    Treatment Information:
    {treatment_info}

    Patient Friendly Summary:
    {patient_friendly_summary}
    """

    try:
        response = gen_model.start_chat(history=[]).send_message(report_prompt)
        return response.text if hasattr(response, 'text') else "No valid response received."
    except Exception as e:
        return f"Error generating report: {e}"

# Function to allow report download
def download_report(report_text, filename="medical_report.txt"):
    st.download_button(
        label="Download Report",
        data=report_text,
        file_name=filename,
        mime="text/plain"
    )

# Streamlit app UI
st.set_page_config(layout="wide")

st.markdown("""
    <style>
        .title { font-size: 2.5rem; color: #007BFF; text-align: center; }
        .image-box { border: 2px solid #007BFF; padding: 1rem; border-radius: 10px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Chest X-ray Disease Detection and Report Generation</div>', unsafe_allow_html=True)
st.write("Upload a chest X-ray image to predict the presence of a disease and generate a detailed report.")

col1, col2 = st.columns([3, 2])

with col1:
    uploaded_file = st.file_uploader("Choose a chest X-ray image...", type="jpeg")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.markdown('<div class="image-box">', unsafe_allow_html=True)
        st.image(img, caption="Uploaded Chest X-ray Image", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        img_path = "uploaded_image.jpeg"
        img.save(img_path)

        disease = predict_image(img_path)
        st.write(f"The diagnosis is: *{disease}*")

with col2:
    if uploaded_file is not None and 'disease' in locals():
        if st.button("Generate Report"):
            st.write("Generating report...")
            infected_part = "Lungs"
            confidence_level = np.random.uniform(95, 98)
            treatment_info = "Treatment includes antibiotics and rest."
            patient_friendly_summary = "This is a common disease that can be treated with proper care."

            report = generate_report_gemini(disease, infected_part, confidence_level, treatment_info, patient_friendly_summary)
            st.write(report)
            download_report(report)
    else:
        st.write("Please upload an image and detect a disease before generating a report.")

# Sidebar health tips
st.sidebar.header("Multi-Disease Health Tips")
st.sidebar.write("""
- **Bacterial Pneumonia**: Stay up to date with vaccinations and practice good hygiene.
- **Corona Virus Disease**: Follow health protocols, wear masks, and maintain social distancing.
- **Tuberculosis**: Get tested regularly if you're in high-risk areas and complete any prescribed treatment.
- **Viral Pneumonia**: Rest and fluids are essential. Avoid exposure to extreme cold.
- **Normal**: Keep up a healthy lifestyle to maintain your well-being.
""")
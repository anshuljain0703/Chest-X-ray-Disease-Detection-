# 🩺 MedScan: Chest X-ray Disease Detection and Report Generation

🏆 **Top 10 Finalist – Innov8-2024**

**MedScan** is an AI-powered diagnostic tool developed by our team – **Siddhant, Mrigank, Abhay, and Anshul**.  
The project was selected among the **top 10 teams** for its innovative contribution to the future of medical diagnostics and patient support systems.

---

## 🌟 About the Project

**MedScan** is a smart healthcare web application that enables users to upload chest X-ray images and receive accurate disease predictions along with downloadable medical reports.  
It integrates deep learning and **Google Gemini-based Generative AI** to support healthcare professionals and enhance early detection for patients.

---

## 🔧 Features & Functionalities

### 🧠 AI-Based Disease Prediction
Detects diseases from chest X-rays with high confidence.  
**Supported classifications:**
- **Bacterial Pneumonia**
- **Viral Pneumonia**
- **COVID-19 (Corona Virus Disease)**
- **Tuberculosis**
- **Normal (Healthy)**

---

### 📋 Automated Report Generation
Generates detailed, patient-friendly reports using the **Google Gemini API**.  
Each report includes:
- Disease Summary  
- Confidence Level  
- Affected Area  
- Recommendations & Advice

---

### 📁 Report Download
Download the diagnosis in `.txt` format  
for personal records or medical consultation.

---

### 💻 Streamlit-Powered Interface
- Intuitive UI for uploading X-ray images  
- Real-time prediction and instant report display

---

## 🚀 Future Enhancements

- 🔬 **More Disease Categories**: Add more thoracic conditions  
- 🌐 **Telemedicine Integration**: Direct consultations with professionals  
- 📱 **Mobile App Version**: On-the-go diagnosis and tracking  
- 🌍 **Multi-language Support**: Break accessibility barriers  
- 🧾 **Medical History Storage**: Secure user login for longitudinal recordkeeping  

---

## 🛠️ Tech Stack

| Component         | Technology                     |
|------------------|--------------------------------|
| **Frontend/UI**  | Streamlit (Python)             |
| **Model**        | VGG16 Deep Learning (Keras)    |
| **Report Gen.**  | Google Gemini API              |
| **Backend**      | Python                         |
| **Environment**  | Jupyter Notebook, Streamlit, VirtualEnv |

---

## 🌐 Live Deployment

🚀 **Try MedScan now on Hugging Face Spaces**  
👉 [Click Here to Launch App](https://huggingface.co/spaces/Anshul-jain07/MedScan)

---

## 📞 Contact

For queries, suggestions, or collaborations:  
📧 **Email**: [anshuljain071103@gmail.com](mailto:anshuljain071103@gmail.com)  
🧑‍💻 **GitHub**: [Anshul-jain07](https://github.com/anshuljain0703)

---

## 💡 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/MedScan.git

# Navigate to the project directory
cd MedScan

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py

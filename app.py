import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# API kalitini kiriting
API_KEY = "sizning_api_kalitingiz"

# API-ga so'rov jo'natish funksiyasi
def detect_objects(image_bytes):
    url = "https://api.api-ninjas.com/v1/objectdetection"
    headers = {'X-Api-Key': API_KEY}
    files = {'image': image_bytes}
    response = requests.post(url, headers=headers, files=files)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("API so'rovi bajarilmadi.")
        return None

# Streamlit interfeys
st.title("Object Detection API Demo")
st.write("Tasvirni yuklang va ob'ektlar avtomatik aniqlanadi.")

# Tasvir yuklash imkoniyati
uploaded_file = st.file_uploader("Tasvir yuklang", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Yuklangan Tasvir', use_column_width=True)
    
    # Tasvirni APIga yuborish
    with st.spinner("Ob'ektlar aniqlanmoqda..."):
        result = detect_objects(uploaded_file)

    # Natijalarni ko'rsatish
    if result:
        st.write("Aniqlangan ob'ektlar:")
        for obj in result:
            st.write(f"Ob'ekt turi: {obj['object']}, Koordinatalari: {obj['box']}")
    else:
        st.write("Hech qanday ob'ekt topilmadi.")

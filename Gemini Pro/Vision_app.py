import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
from PIL import Image

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

def get_response(input, image):
    if input!="":
        response = model.generate_content([input, image])
    else:
        response = model.generate_content(image)
    return response.text

st.title("Image Describer")
input = st.text_input("Input : ", key = "input")
submit = st.button("Describe..")

uploaded_file = st.file_uploader("Please upload an Image", type = ["png", "jpeg", "jpg"])
image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption = "Uploaded Image", use_container_width = True)

if submit:
    response = get_response(input, image)
    st.write(response)
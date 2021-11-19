import numpy as np
import streamlit as st
from PIL import Image

from app import ClassifyWear

from PIL import Image # Required to show images
import pandas as pd
logo = Image.open("data/logo.png")
st.sidebar.image(logo, width=250)

# Text/Title
st.title("Fashion Prediction ")

st.sidebar.header("Jack & Jones Team")
st.sidebar.text("Team members")
st.sidebar.write("""
# Anthony
# Sai
# Harsha""")

model_pth = 'data/best_model.pt'
path = None
predicted = None
prob = None


uploaded_file = st.file_uploader('Upload Image File')
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, channels="BGR", width=100)
    prob, predicted = ClassifyWear(model_pth).predict(uploaded_file)


if predicted is not None:
    st.write(f'The image passed is {predicted} with a probability of {prob}%')



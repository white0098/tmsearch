import streamlit as st
import pandas as pd
import numpy as np

st.title('TMSEARCH')

# input trademark image
st.header('Trademark Image')
img_file_buffer = st.file_uploader("Upload an trademark image", type=["png", "jpg", "jpeg"])

# input text


# viewer
if img_file_buffer is not None:
    st.image(img_file_buffer)

# text input
st.header('Trademark Text')
text_input = st.text_input('Enter trademark text', 'Type here...')
st.write('The current text is', text_input)

import streamlit as st
import pandas as pd
import numpy as np

from main import tmsearch

st.title("TMSEARCH DEMO")


# two column layout
col1, col2 = st.columns(2)


# input trademark image


with col1:
    img_file1_buffer = st.file_uploader(
        "Upload trademark image 1", type=["png", "jpg", "jpeg"]
    )

    if img_file1_buffer is not None:
        img_file1 = img_file1_buffer.read()
        st.image(img_file1, caption="Uploaded Image.", use_column_width=True)
        with open("tmp1.jpg", "wb") as f:
            f.write(img_file1)


with col2:
    img_file2_buffer = st.file_uploader(
        "Upload Trademark image 2", type=["png", "jpg", "jpeg"]
    )

    if img_file2_buffer is not None:
        img_file2 = img_file2_buffer.read()
        st.image(img_file2, caption="Uploaded Image.", use_column_width=True)
        with open("tmp2.jpg", "wb") as f:
            f.write(img_file2)


if st.button("유사도 측정", key="유사도 측정",use_container_width=True, help="유사도 측정 버튼"):
    result_dict = tmsearch("tmp1.jpg", "tmp2.jpg")
    st.write(result_dict)
    
    
    
    

    
    







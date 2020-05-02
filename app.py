import streamlit as st
import numpy as np
import time
from prepare import load_ml_model
from PIL import Image
import io

st.title("CoViD19 - Image Classification")

st.sidebar.title("Input Tuning")
st.sidebar.markdown("#### Upload an image")
file_ = st.sidebar.file_uploader("", ["png", "jpg", "jpeg"], encoding=None)
model = load_ml_model()

if file_ is not None:
    st.image(file_, use_column_width=True)
    with st.spinner("Classifying Image ..."):
        temporarylocation = "aa"
        with open(temporarylocation, "wb") as out:  ## Open temporary file as bytes
            out.write(file_.read())

        img = Image.open(temporarylocation)
        img = img.convert("LA").resize((331, 331))
        img = np.asanyarray(img)[:, :, 0].flatten()
        res = model.predict_proba(np.asanyarray([img]))[0][0]

        st.markdown(
            f"### This is an image of: Covid19 ({(res*100):0.2f}%) and Flu Virus ({((1-res)*100):0.2f}%)"
        )

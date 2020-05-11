import streamlit as st
from PIL import Image
import numpy as np
import time
import helpers
import model
import io
import os
from threading import Thread

st.title("CoViD19 - Image Classification")

st.sidebar.title("Input Tuning")
st.sidebar.markdown("#### Upload an image")
file_ = st.sidebar.file_uploader("", ["png", "jpg", "jpeg"], encoding=None)
ml_model = model.load_ml_model(model.ML_MODEL_FILE)

if file_ is not None:
    st.image(file_, use_column_width=True)
    with st.spinner("Classifying Image ..."):
        temporarylocation = time.strftime("%Y%m%d_%H%M%S")
        with open(temporarylocation, "wb") as out:  ## Open temporary file as bytes
            out.write(file_.read())

        img = Image.open(temporarylocation)
        img = img.convert("LA").resize((331, 331))
        img.save(temporarylocation + ".png")
        img = np.asanyarray(img)[:, :, 0].flatten()
        res = ml_model.predict_proba(np.asanyarray([img]))[0][0]
        
        st.markdown(
            f"### This is an image of: Covid19 ({(res*100):0.2f}%) and Flu Virus ({((1-res)*100):0.2f}%)"
        )

        yes = st.button("True")
        no = st.button("False")

        if yes or no:
            process = Thread(target=model.train_model)
            process.start()
            if yes:
                helpers.move_image(temporarylocation + ".png", res >= 0.5)
            elif no:
                helpers.move_image(temporarylocation + ".png", res < 0.5)
            st.markdown("<style>        button {visibility: hidden;} </style>", unsafe_allow_html=True)
        os.remove(temporarylocation)
            
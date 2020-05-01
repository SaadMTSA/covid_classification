import streamlit as st
import numpy as np
import time

st.title("CoViD19 - Image Classification")

st.sidebar.title("Input Tuning")
st.sidebar.markdown("#### Upload an image")
file = st.sidebar.file_uploader("", ["png", "jpg", "jpeg"], encoding=None)

if file is not None:
    st.image(file, use_column_width=True)

    with st.spinner("Classifying Image ..."):
        time.sleep(5)

    probs = np.zeros(5)
    for i in range(1, 6):
        val = np.random.rand()
        probs[i - 1] = val
        if val < 0.5:
            st.success(f"Model #{i} - Negative ({(1-val)*100:0.2f}%)")
        else:
            st.error(f"Model #{i} - Positive ({val*100:0.2f}%)")

    if sum(probs < 0.5) > 2:
        st.markdown("**Most models agree that this is a negative sample.**")
    else:
        st.markdown("**Most models agree that this is a positive sample.**")

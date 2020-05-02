import streamlit as st
import pickle
from keras.models import load_model


@st.cache(allow_output_mutation=True)
def load_ml_model():
    return pickle.load(open("rf.pickle", "rb"))

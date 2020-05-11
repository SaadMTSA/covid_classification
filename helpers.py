import streamlit as st
import pickle
import os
from pathlib import Path


def move_image(image, cls):
    class_ = "covid19"
    if not cls:
        class_ = "flu"

    os.rename(image, Path(f"images/{class_}") / image)

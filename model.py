import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pickle

ML_MODEL_FILE = "rf.pickle"


@st.cache(allow_output_mutation=True)
def load_ml_model(path):
    return pickle.load(open(path, "rb"))


def save_ml_model(model, path):
    pickle.dump(model, open(path, "wb"))


def train_model():
    img_size = (331, 331)

    def create_transforms(image):
        yield image.rotate(45)
        yield image.rotate(90)
        yield image.rotate(135)
        yield image.rotate(180)
        yield image.rotate(225)
        yield image.rotate(270)
        yield image.rotate(315)
        yield ImageOps.mirror(image)
        yield ImageOps.flip(image)

    img_dir = Path("images/")
    output_dir = Path("processed_images/")
    
    try:
        os.mkdir(img_dir)
        os.mkdir(img_dir / "flu")
        os.mkdir(img_dir / "covid19")
        os.mkdir(output_dir)
    except:
        pass

    cls_imgs = {}

    cnt = 0
    for cls in os.listdir(img_dir):
        cls_imgs[cls] = []
        cls_dir = img_dir / cls
        for img in os.listdir(cls_dir):
            image = Image.open(cls_dir / img).convert("LA").resize(img_size)
            transforms = create_transforms(image)
            for i, transforms_i in enumerate(transforms):
                plt.imsave(
                    output_dir / (f"{cls}_{i}_{img}"),
                    np.asanyarray(transforms_i)[:, :, 0],
                )
                cls_imgs[cls].append(np.asanyarray(transforms_i)[:, :, 0].flatten())
                cnt += 1

    x = np.asanyarray([i for _, v in cls_imgs.items() for i in v])
    class2idx = {cls: idx for idx, cls in enumerate(set(cls_imgs.keys()))}
    idx2class = {idx: cls for cls, idx in class2idx.items()}
    y = np.asanyarray([class2idx[k] for k, v in cls_imgs.items() for i in v])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=11, test_size=0.3
    )

    rf = RandomForestClassifier(random_state=11)
    rf.fit(x_train, y_train)

    save_ml_model(rf, ML_MODEL_FILE)

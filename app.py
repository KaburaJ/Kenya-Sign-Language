import os
import streamlit as st
import numpy as np
import PIL.Image
#from PIL import Image
from fastai.vision.all import * 
import pathlib

import matplotlib.pyplot as pt

import pathlib
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

model = load_learner('ksl_model.pkl')

def predict(image_path):
    # load the image and convert into
    # numpy array
    #image= Image.open(image)
    # image = Image.open(image)
    # PIL images into NumPy arrays
    pred_label= model.predict(image_path)

    return pred_label


def show_likelihood(pred_label):
    class_probs = pred_label[2].numpy()
    classes = ["Temple", "You", "Me", "You", "Friend", "Love", "Enough", "Church","Mosque"]
    class_labels = [classes[i] for i in range(len(class_probs))]
    fig = pt.figure(figsize=(10, 10))
    pt.barh(class_labels, class_probs)
    pt.ylabel("Class")
    pt.xlabel("Probability")
    pt.title("Class Probabilities")
    pt.xlim(0, 1)
    pt.ylim(-1, len(class_probs))
    st.pyplot(fig)

def main():
    st.image('ksl1.jpg')

    st.write("# KSL Image Classification App")
    st.write("This app allows you to upload an image and have it classified by a trained machine learning model.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:

        image = PIL.Image.open(uploaded_file)

        image_path = os.path.join("tempDir",uploaded_file.name)

        with open(image_path, "wb") as f: 
          f.write(uploaded_file.getbuffer())

        st.image(image, caption="Uploaded Image", use_column_width=True)
        pred_label = predict(image_path)
        st.write("The image was classified as:", pred_label[0])

        show_likelihood(pred_label)

if __name__ == '__main__':
    main()

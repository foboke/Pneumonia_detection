import streamlit as st 
import tensorflow as tf 
import numpy as np 
from PIL import Image, ImageOps


st.write(""" # Pneumonia detection on Chest X-Rray """)
upload_file = st.sidebar.file_uploader("Upload Chest X-Ray", type=['jpg','png','jpeg'])
Generate_pred=st.sidebar.button("Diagnostic")
model=tf.keras.models.load_model('useful_files/model_pneumonia_new.h5')


def preprocessed_image(file):
    image = file.resize((44,44), Image.ANTIALIAS)
    image = np.array(image)
    image = np.expand_dims(image, axis=0) 
    return image


def import_n_pred(image_data, model):
    size = (150,150)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    reshape=img[np.newaxis,...]
    pred = model.predict(reshape)
    return pred
  
  
if Generate_pred:
    image=Image.open(upload_file)
    with st.expander('X-Ray', expanded = True):
        st.image(image, use_column_width=True)
    #pred=import_n_pred(image, model)
    used_images = preprocessed_image(image)
    labels = ['Pneumonia', 'Sain']
    #st.title("Prediction of image is {}".format(labels[np.argmax(pred)]))
    
    predictions = np.argmax(model.predict(used_images), axis=-1)
    if predictions == 1:
        st.error("Cells get parasitized")
    elif predictions == 0:
        st.success("Cells is healty Uninfected")

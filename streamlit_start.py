import streamlit as st 
import tensorflow as tf 
import numpy as np 
from PIL import Image, ImageOps


st.write(""" # Pneumonia detection on Chest X-Rray """)
upload_file = st.sidebar.file_uploader("Upload Chest X-Ray", type="jpeg")
Generate_pred=st.sidebar.button("Diagnostic")
model=tf.keras.models.load_model('useful_files/model_pneumonia.h5')

def import_n_pred(image_data, model):
    size = (150,150)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    reshape=img[np.newaxis,...]
    pred = model.predict(reshape)
    return pred
  
  
if Generate_pred:
    image=Image.open(upload_file)
    with st.beta_expander('X-Ray', expanded = True):
        st.image(image, use_column_width=True)
    pred=import_n_pred(image, model)
    labels = ['Parasitized', 'Uninfected']
    st.title("Prediction of image is {}".format(labels[np.argmax(pred)]))

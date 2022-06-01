import streamlit as st 
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf 
import numpy as np 
from PIL import Image, ImageOps


st.write(""" # Pneumonia detection on Chest X-Rray """)
upload_file = st.sidebar.file_uploader("Upload Chest X-Ray", type=['jpg','png','jpeg'])
Generate_pred=st.sidebar.button("Diagnostic")
model=tf.keras.models.load_model('useful_files/model_pneumonia_new.h5')


def preprocessed_image(file):
    image = file.resize((150,150))
    image = np.array(image)
    image = np.expand_dims(image, axis=0) 
    return image


def import_n_pred(image_data, model):
    size = (150,150, 3)
    image = ImageOps.fit(image_data, size )
    img = np.asarray(image)
    reshape=img[np.newaxis,...]
    pred = model.predict(reshape)
    return pred
  
    
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="The Person is Infected With Pneumonia"
    else:
        preds="The Person is not Infected With Pneumonia"
    
    
    return preds
    
  
if Generate_pred:
    the_image=Image.open(upload_file)
    with st.expander('X-Ray', expanded = True):
        st.image(the_image, use_column_width=True)
    #pred=import_n_pred(image, model)
    
    #predictions = model_predict( the_image, model)
    used_images = preprocessed_image(the_image)
    labels = ['Pneumonia', 'Sain']
    #st.title("Prediction of image is {}".format(labels[np.argmax(pred)]))
    
    predictions = np.argmax(model.predict(used_images), axis=-1)
    if predictions == 1:
        st.error("Cells get parasitized")
    elif predictions == 0:
        st.success("Cells is healty Uninfected")

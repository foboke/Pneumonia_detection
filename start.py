import streamlit as st
import os
import sys

st.sidebar.title("Pneumonia detection Classifier Interface ")
PATH_TO_TEST_IMAGES = './test_images/'

def get_list_of_images():
    file_list = os.listdir(PATH_TO_TEST_IMAGES)
    return [str(filename) for filename in file_list if str(filename).endswith('.jpg')]



def main():
    st.sidebar.subheader('Load image')
    image_file_uploaded = st.sidebar.file_uploader('Upload an image', type = 'png')
    st.sidebar.text('OR')
    image_file_chosen = st.sidebar.selectbox('Select an existing image:', get_list_of_images())
    
   


if __name__ == '__main__':
    main()

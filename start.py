import streamlit as st
import os
import sys

st.sidebar.title("Pneumonia detection Classifier Interface ")




def main():
    st.sidebar.subheader('Load image')
    image_file_uploaded = st.sidebar.file_uploader('Upload an image', type = 'jpg')
    
    
    
   


if __name__ == '__main__':
    main()

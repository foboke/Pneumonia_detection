import streamlit as st


def load_image():
    uploaded_file = st.file_uploader(label='Upload the image of the chest X-ray')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)


def main():
    st.title('Pneumonia detection interface')
    load_image()


if __name__ == '__main__':
    main()

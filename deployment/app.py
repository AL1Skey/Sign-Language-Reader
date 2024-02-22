import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow.keras as keras
import os
from PIL import Image

from inference import image_preprocessing, model_pred

model = keras.models.load_model('seq_modelv2.hdf5')

# Streamlit web app
def main():
    st.title("Sign Language App")

    uploaded_file = st.file_uploader("Choose a file",type=(['png','jpg','jpeg']))
    
    if uploaded_file is not None:
        
        st.subheader("Uploaded File")
        st.image(uploaded_file)
        uploaded_file = Image.open(uploaded_file)
        image = image_preprocessing(uploaded_file)
        words = model_pred(model,image)
        
        # See prediction
        st.write(f'# Model See this Sign Language as "{words}"')

if __name__ == "__main__":
    main()






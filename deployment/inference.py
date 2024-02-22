import tensorflow as tf
import numpy as np
import string
from tensorflow.keras.preprocessing.image import load_img

def image_preprocessing(data):
    data = np.array(data)
    image_array= tf.keras.preprocessing.image.img_to_array(data)

    # Resize the image
    new_image = tf.image.resize(image_array,(32,32))
    new_image = tf.expand_dims(new_image, axis=0)
    new_image = np.array(new_image)
    
    return new_image

def model_pred(model,feature):
    pred = model.predict(feature)
    index = np.argmax(pred) if pred[0][np.argmax(pred)]>0.5 else 26
    dec = {i:j for i,j in enumerate(string.ascii_lowercase)}
    
    if index >= len(dec):
        dec[index] = 'Unknown' 
    
    index = dec[index]
    
    return index
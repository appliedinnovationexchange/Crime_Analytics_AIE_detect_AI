import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
# Function to load and prepare the image
def load_and_prepare_image(image, target_size=(256, 256)):
    # Convert the image to RGB
    image = image.convert('RGB')
    # Resize the image
    image = image.resize(target_size)
    # Convert the image to a numpy array
    image_array = img_to_array(image)
    # Scale from [0, 255] to [-1, 1]
    image_array = (image_array - 127.5) / 127.5
    # Add a batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array
# Load your trained model
model = load_model('perumal_100_data_saved_model.h5')
# Streamlit UI
st.title("Criminal Detection App")
# File uploader allows the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    # Display the uploaded image
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    # Prepare the image
    prepared_image = load_and_prepare_image(uploaded_image)
    # Generate the image
    generated_image = model.predict(prepared_image)
    # Scale from [-1, 1] to [0, 255]
    generated_image = (generated_image + 1) / 2.0
    generated_image = np.squeeze(generated_image) * 255.0
    # Convert the generated image to a PIL Image
    generated_image = Image.fromarray(generated_image.astype('uint8'))
    # Display the generated image
    st.image(generated_image, caption='Generated Image', use_column_width=True)
 
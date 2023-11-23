import streamlit as st
import numpy as np
from keras.preprocessing import image as keras_image
from keras.models import load_model
import requests
from bs4 import BeautifulSoup

# Load the model (Replace 'FV.h5' with your actual model file)
model = load_model('FV.h5')

# Define class labels
labels = {
    0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
    7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
    14: 'grapes', 15: 'jalapeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
    19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
    26: 'pomegranate', 27: 'potato', 28: 'radish', 29: 'soybeans', 30: 'spinach', 31: 'sweetcorn',
    32: 'sweet potato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'
}

# Define categories
fruits = ['apple', 'banana', 'grapes', 'kiwi', 'lemon', 'mango', 'orange', 'paprika', 'pear', 'pineapple', 'pomegranate', 'watermelon']
vegetables = ['beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'jalapeno', 'lettuce', 'onion', 'peas', 'potato', 'radish', 'soybeans', 'spinach', 'sweetcorn', 'sweet potato', 'tomato', 'turnip']

# Function to fetch nutrient information from Google search
def fetch_nutrient_info(prediction, nutrient):
    try:
        url = f'https://www.google.com/search?q={nutrient} in {prediction}'
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        nutrient_info = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return nutrient_info
    except Exception as e:
        st.error(f"Can't able to fetch {nutrient} information")
        print(e)

# Function to process the uploaded image
def processed_img(img_path):
    try:
        img = keras_image.load_img(img_path, target_size=(224, 224))
        img = keras_image.img_to_array(img)
        img = img / 255
        img = np.expand_dims(img, axis=0)
        answer = model.predict(img)
        y_class = answer.argmax(axis=-1)
        y = int(y_class)
        res = labels[y]
        return res.capitalize()
    except Exception as e:
        st.error("Error processing the image. Please make sure it's a valid image file.")
        print(e)
        return "Unknown"

# Streamlit app
def run():
    st.title("Food Ingredient Analysis")
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        img = st.image(img_file, use_column_width=False, caption="Uploaded Image")

        # Get the predicted category
        result = processed_img(img_file)
        
        # Determine if it's a fruit or vegetable
        if result.lower() in fruits:
            st.info('**Category : Fruits🍍**')
        elif result.lower() in vegetables:
            st.info('**Category : Vegetables🍅**')
        else:
            st.warning('**Category : Unknown**')

        # Display the predicted category
        st.success("**Predicted : " + result + '**')

        # Fetch and display nutrients (if available)
        nutrients = ['Calories', 'Protein', 'Carbohydrate', 'Sugar', 'Fat', 'Vitamin', 'Fiber', 'Cholesterol']
        for nutrient in nutrients:
            nutrient_info = fetch_nutrient_info(result, nutrient)
            if nutrient_info:
                st.warning(f'**{nutrient}: {nutrient_info} (per 100 grams)**')

if __name__ == "__main__":
    run()

# Food-Ingredient-Analysis-Using-TensorFlow
## Overview

This is a Streamlit web application for food ingredient analysis. The app uses a pre-trained neural network model to predict the category (fruit or vegetable) of a given food image. Additionally, it attempts to fetch nutrient information for the predicted food item through a Google search.

## Prerequisites

- Python 3.6 or later
- Required Python packages can be installed using the following command:
  ```bash
  pip install -r requirements.txt

## Setup
Clone the repository:
git clone https://github.com/satyadeep-2308/Food-Ingredient-Analysis-Using-TensorFlow.git
cd Food-Ingredient-Analysis-Using-TensorFlow

## Install dependencies:
pip install -r requirements.txt

## Run the app:
streamlit run app.py
Open the provided URL in your web browser to access the application.

## Usage
Upload an image using the "Choose an Image" button.
The app will display the uploaded image and predict whether it belongs to the "Fruits" or "Vegetables" category.
Additional nutrient information (if available) will be fetched and displayed.

## Model
The neural network model (FV.h5) used for predictions is pre-trained on a dataset containing various fruits and vegetables.

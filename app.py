import streamlit as st
import cv2
import pandas as pd
import numpy as np
import re
import os
from PIL import Image
from paddleocr import PaddleOCR

# Initialize PaddleOCR
@st.cache_resource
def load_ocr():
  return PaddleOCR(use_angle_cls=True, lang='en')

ocr = load_ocr()

# Function to preprocess the image for better OCR performance
def preprocess_image(image):
  # Convert to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Denoising for better OCR
  gray = cv2.fastNlMeansDenoising(gray, h=30)

  # Adaptive thresholding for better text recognition
  thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

  return thresh

# Function to extract text from an image using PaddleOCR
def extract_text_from_image(image):
  # Preprocess the image
  processed_image = preprocess_image(image)
  
  try:
      # Extract text using PaddleOCR
      result = ocr.ocr(processed_image)
      text = ' '.join([line[1][0] for line in result])
      return text
  except Exception as e:
      st.error(f"Error in OCR: {str(e)}")
      return None

# Function to parse relevant information from extracted text
def parse_boiler_data(text):
  boiler_data = {}

  # Use regex to find power rating (e.g., '200 kW')
  power_rating_match = re.search(r'(\d+)\s?kW', text)
  if power_rating_match:
      boiler_data['Power Rating (kW)'] = int(power_rating_match.group(1))

  # Find model number (replace regex with the correct format for your data)
  model_match = re.search(r'Model\s?:\s?([A-Za-z0-9\-]+)', text, re.IGNORECASE)
  if model_match:
      boiler_data['Model'] = model_match.group(1)

  return boiler_data

# Function to estimate energy use
def calculate_energy_use(power_rating_kw, hours_per_day, days_per_year=365):
  return power_rating_kw * hours_per_day * days_per_year

# Function to save extracted data into a CSV
def save_to_csv(data, filename="boiler_assets.csv"):
  if not os.path.isfile(filename):
      df = pd.DataFrame([data])
      df.to_csv(filename, mode='w', index=False, header=True)
  else:
      df = pd.DataFrame([data])
      df.to_csv(filename, mode='a', index=False, header=False)

# Streamlit App Interface
st.title('Boiler Nameplate OCR and Energy Estimator')

# File uploader for the nameplate image
uploaded_file = st.file_uploader("Upload Boiler Nameplate Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
  # Convert the uploaded image to an OpenCV format
  image = np.array(Image.open(uploaded_file))
  
  # Display the uploaded image
  st.image(image, caption="Uploaded Nameplate", use_column_width=True)

  # Extract text from the image
  extracted_text = extract_text_from_image(image)
  if extracted_text:
      st.subheader("Extracted Text")
      st.text(extracted_text)

      # Parse the extracted text
      boiler_data = parse_boiler_data(extracted_text)

      # Display parsed data and allow manual overrides
      st.subheader("Boiler Information")
      power_rating = st.number_input("Power Rating (kW)", value=boiler_data.get('Power Rating (kW)', 0))
      model_number = st.text_input("Model Number", value=boiler_data.get('Model', ''))

      operational_hours = st.number_input("Operational Hours per Day", value=8, min_value=0, max_value=24)
      operational_days = st.number_input("Operational Days per Year", value=365, min_value=0, max_value=365)

      # Calculate energy use
      if power_rating > 0:
          estimated_energy_use = calculate_energy_use(power_rating, operational_hours, operational_days)
          st.subheader(f"Estimated Energy Use: {estimated_energy_use:.2f} kWh/year")
          
          # Save data to CSV
          if st.button('Save to CSV'):
              boiler_data['Power Rating (kW)'] = power_rating
              boiler_data['Model'] = model_number
              boiler_data['Operational Hours (per day)'] = operational_hours
              boiler_data['Operational Days (per year)'] = operational_days
              boiler_data['Energy Use (kWh/year)'] = estimated_energy_use
              
              save_to_csv(boiler_data)
              st.success("Boiler data saved to CSV.")

else:
  st.info("Please upload a boiler nameplate image to get started.")

# Created/Modified files during execution:
print("boiler_assets.csv")

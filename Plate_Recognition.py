import streamlit as st
import sys
import pickle
import cv2
import numpy as np

# Remove pytesseract import and configuration since the pipeline now uses EasyOCR
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Add the YOLOv5 repository path to sys.path so that the 'models' module is found
sys.path.insert(0, r"C:\Users\user\Desktop\yolov5\yolov5")

# Import the custom pipeline class from the separate module.
from plate_pipeline import LicensePlateRecognizer

# Load the pickled pipeline object. 
# Ensure that this pickle file corresponds to the updated version of your pipeline.
pickle_path = r'C:\Users\user\Desktop\license_plate_recognizer.pkl'
with open(pickle_path, 'rb') as f:
    recognizer = pickle.load(f)

st.title("License Plate Recognition App 🚗🔍")
st.write("Upload an image, and the app will detect if it contains a license plate. If detected, it will extract the plate details.")

# File uploader for image input.
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Convert the uploaded file to a NumPy array.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)  # Read image as BGR

    if image is None:
        st.error("Unable to read the image. Please try a different image file.")
    else:
        # Display the uploaded image.
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

        # Run detection on the image.
        plate_roi, detections = recognizer.detect_plate(image)

        # Check if any detection exists.
        if detections is None or len(detections) == 0:
            st.write("No license plate detected in the image.")
        else:
            # Run OCR on the cropped license plate area using EasyOCR (inside the updated recognizer).
            plate_text = recognizer.recognize_plate(plate_roi)
            
            if plate_text:
                st.write("**Detected License Plate Text:**", plate_text)
            else:
                st.write("License plate detected, but no text was extracted.")

            # Optionally, display the detected plate region.
            st.image(cv2.cvtColor(plate_roi, cv2.COLOR_BGR2RGB), caption="Detected License Plate ROI", use_column_width=True)

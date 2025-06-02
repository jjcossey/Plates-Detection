License Plate Recognition Using YOLOv5 & OCR

An AI-powered system for automatic vehicle license plate detection and recognition.
Overview
This project aims to develop an efficient License Plate Recognition (LPR) system using YOLOv5 for plate detection and Optical Character Recognition (OCR) for extracting plate numbers. Designed for real-world applications such as security surveillance, automated toll systems, and smart parking solutions, this model accurately identifies and reads vehicle license plates from images or video feeds.
Features
Real-Time Detection – Fast and accurate recognition using YOLOv5.

OCR Integration – Extracts and processes plate numbers efficiently.

Custom Dataset – Trained on carefully annotated license plate images.

High Accuracy – Optimized model with fine-tuned hyperparameters.

Scalable Deployment – Supports integration into web and mobile applications.

Technology Stack

YOLOv5 – Deep learning model for object detection.

Easy OCR – Extracting text from detected plates.

Python – Core programming language for implementation.

OpenCV – Image processing and video frame extraction.

TensorFlow / PyTorch – Model training and inference.

Streamlit (Optional) – For containerized deployment.

Installation & Usage
Step 1: Clone the Repository
git clone https://github.com/your-username/license-plate-recognition.git
cd license-plate-recognition

Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Run the Detection Model
Streamlit run Plate_Recognition.py

Dataset & Training
Custom annotated dataset of license plates.
Dataset was obtained from kaggle i.e can be accessed throught the following link
https://www.kaggle.com/datasets/andrewmvd/car-plate-detection

Augmented images for better generalization.

Training using YOLOv5 with fine-tuned hyperparameters.



Feel free to contribute by improving the dataset, optimizing the model, or integrating additional features. Open a pull request or reach out via GitHub Issues.

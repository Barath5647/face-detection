import gdown
import zipfile
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import torch
import streamlit as st
import datetime

# Download the dataset CSV and images folder from Google Drive using gdown

# URL for your CSV file
csv_url = 'https://drive.google.com/uc?id=1-COHZDOHOUbfosASSh-MJdnhP-yhPhln&export=download
'
gdown.download(csv_url, 'faces.csv', quiet=False)

# URL for your images folder (assuming it's zipped)
image_folder_url = 'https://drive.google.com/uc?id=1EJOmT0WxTITiMFYb5n-71UYQTwaZfmDC&export=download
'
gdown.download(image_folder_url, 'images.zip', quiet=False)

# Step 3: Unzip images if it's a ZIP file
zip_path = 'images.zip'
extract_dir = './images/'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Step 4: Load the dataset (CSV)
data = pd.read_csv('faces.csv')
image_dir = extract_dir  # The path where images were extracted

# Step 5: Define your model (e.g., YOLOv5)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Using YOLOv5 small model

# Preprocessing and feature extraction functions remain the same

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (224, 224))

def detect_faces(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)
    return results.xyxy[0].cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)

# Real-time detection and logging functions remain the same

# Streamlit app UI
st.title("Human Face Detection System")
st.sidebar.title("Options")
use_case = st.sidebar.selectbox("Choose a Business Use Case",
                                ["Security", "Access Control", "Retail",
                                 "Healthcare", "Automotive", "Entertainment"])

option = st.sidebar.selectbox("Choose Functionality", ["Static Image Detection", "Real-time Detection", "Model Evaluation"])

if use_case == "Security":
    st.write("Monitoring public spaces...")
    if option == "Static Image Detection":
        image_name = st.sidebar.text_input("Enter Image Name:")
        if image_name:
            image_path = os.path.join(image_dir, image_name)  # Set the correct image path
            detections = detect_faces(image_path)
            st.image(image_path, caption="Input Image")
            st.write("Detections:", detections)

    elif option == "Real-time Detection":
        st.write("Starting real-time detection...")
        # Real-time detection code here

elif use_case == "Access Control":
    st.write("Secure access to buildings and devices...")
    if option == "Static Image Detection":
        image_name = st.sidebar.text_input("Enter Image Name:")
        if image_name:
            image_path = os.path.join(image_dir, image_name)  # Set the correct image path
            detections = detect_faces(image_path)
            st.image(image_path, caption="Input Image")
            st.write("Detections:", detections)

elif use_case == "Retail":
    st.write("Analyzing customer emotions...")
    if option == "Static Image Detection":
        image_name = st.sidebar.text_input("Enter Image Name:")
        if image_name:
            image_path = os.path.join(image_dir, image_name)  # Set the correct image path
            # Emotion analysis code here

elif use_case == "Healthcare":
    st.write("Tracking patient conditions...")
    if option == "Static Image Detection":
        image_name = st.sidebar.text_input("Enter Image Name:")
        if image_name:
            image_path = os.path.join(image_dir, image_name)  # Set the correct image path
            detections = detect_faces(image_path)
            st.image(image_path, caption="Input Image")
            st.write("Detections:", detections)

elif use_case == "Automotive":
    st.write("Monitoring driver attention and safety...")
    if option == "Real-time Detection":
        st.write("Starting real-time driver monitoring...")
        # Real-time detection code here

elif use_case == "Entertainment":
    st.write("Enhancing gaming and virtual reality experiences...")
    if option == "Real-time Detection":
        st.write("Starting real-time VR enhancement...")
        # Real-time detection code here

elif option == "Model Evaluation":
    y_true = st.text_input("Enter Ground Truth Labels (comma-separated):")
    y_pred = st.text_input("Enter Predicted Labels (comma-separated):")
    if y_true and y_pred:
        y_true = list(map(int, y_true.split(',')))
        y_pred = list(map(int, y_pred.split(',')))
        precision, recall, f1, avg_precision = evaluate_model(y_true, y_pred)
        st.write(f"Precision: {precision}")
        st.write(f"Recall: {recall}")
        st.write(f"F1 Score: {f1}")
        st.write(f"Average Precision: {avg_precision}")

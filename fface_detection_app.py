# Import necessary libraries
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import dlib
from skimage.feature import hog, local_binary_pattern
from skimage import exposure
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import torch
import streamlit as st
import datetime

import torch  # PyTorch is required to load the model

# Load the YOLOv5 pre-trained model from the Ultralytics repository
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s' is the smallest version of YOLOv5
# Import the necessary library from ultralytics
from ultralytics import YOLO

# Load the YOLOv5 pre-trained model (adjust the model size as needed)
model = YOLO("yolov5s.pt")  # or yolov5m.pt or yolov5l.pt depending on your choice

import pandas as pd

def load_dataset(csv_path, image_dir):
    data = pd.read_csv(csv_path)
    return data, image_dir

csv_path = '/content/drive/MyDrive/faces/faces.csv'
image_dir = '/content/drive/MyDrive/faces/images'
data, images_path = load_dataset(csv_path, image_dir)

print(data.head())
print("Images Path:", images_path)

# Step 2: Preprocessing
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Resize(224, 224),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (224, 224))

# Step 3: Feature Extraction
def extract_features(image_path):
    image = preprocess_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # HOG features
    hog_features = hog(gray, visualize=False)
    # LBP features
    lbp_features = local_binary_pattern(gray, 8, 1, method='uniform')
    return hog_features, lbp_features

# Step 4: Face Detection
def detect_faces(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)
    return results.xyxy[0].cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)

# Step 5: Real-time Detection
def real_time_detection():
    cap = cv2.VideoCapture(0)  # Open webcam
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        results.show()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Step 6: Log Detection (for security and access control)
def log_detection(detections):
    with open("detection_log.csv", "a") as log_file:
        for det in detections:
            log_file.write(f"{det}, {datetime.datetime.now()}\n")

# Step 7: Match Face (for access control)
def match_face(detected_face, authorized_faces):
    for auth_face in authorized_faces:
        if compare_faces(detected_face, auth_face):  # Use face embeddings
            return True
    return False

# Step 8: Emotion Analysis (for retail)
def analyze_emotions(image):
    emotion_model = load_emotion_model()  # Load a pre-trained model
    emotions = emotion_model.predict(image)
    return emotions

# Step 9: Driver Attention Monitoring (for automotive)
def monitor_driver(frame):
    detections = model(frame)
    if "closed_eyes" in detections:
        alert_driver()  # Implement an alert for driver

# Step 10: Model Evaluation
def evaluate_model(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred)
    return precision, recall, f1, avg_precision

# Step 11: Streamlit UI
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
            image_path = os.path.join(images_path, image_name)
            detections = detect_faces(image_path)
            log_detection(detections)  # Log the detections
            st.image(image_path, caption="Input Image")
            st.write("Detections:", detections)

    elif option == "Real-time Detection":
        st.write("Starting real-time detection...")
        real_time_detection()

elif use_case == "Access Control":
    st.write("Secure access to buildings and devices...")
    if option == "Static Image Detection":
        image_name = st.sidebar.text_input("Enter Image Name:")
        if image_name:
            image_path = os.path.join(images_path, image_name)
            detections = detect_faces(image_path)
            authorized_faces = []  # Load authorized faces from a database
            if match_face(detections, authorized_faces):
                st.write("Access Granted")
            else:
                st.write("Access Denied")
            st.image(image_path, caption="Input Image")
            st.write("Detections:", detections)

elif use_case == "Retail":
    st.write("Analyzing customer emotions...")
    if option == "Static Image Detection":
        image_name = st.sidebar.text_input("Enter Image Name:")
        if image_name:
            image_path = os.path.join(images_path, image_name)
            emotions = analyze_emotions(image_path)
            st.image(image_path, caption="Input Image")
            st.write("Detected Emotions:", emotions)

elif use_case == "Healthcare":
    st.write("Tracking patient conditions...")
    if option == "Static Image Detection":
        image_name = st.sidebar.text_input("Enter Image Name:")
        if image_name:
            image_path = os.path.join(images_path, image_name)
            detections = detect_faces(image_path)
            # Use additional models to detect distress or other conditions
            st.image(image_path, caption="Input Image")
            st.write("Detections:", detections)

elif use_case == "Automotive":
    st.write("Monitoring driver attention and safety...")
    if option == "Real-time Detection":
        st.write("Starting real-time driver monitoring...")
        real_time_detection()

elif use_case == "Entertainment":
    st.write("Enhancing gaming and virtual reality experiences...")
    if option == "Real-time Detection":
        st.write("Starting real-time VR enhancement...")
        real_time_detection()

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

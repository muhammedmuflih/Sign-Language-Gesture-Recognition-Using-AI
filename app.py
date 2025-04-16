import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import os
from collections import deque

# Configuration
MODEL_PATH = "sign_language_model.keras"
CLASS_NAMES_PATH = "class_names.npy"
ROI_SIZE = 300
IMG_SIZE = 64
HISTORY_SIZE = 5
MIN_CONFIDENCE = 0.85

# Cache resources
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please train the model first.")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_class_names():
    if not os.path.exists(CLASS_NAMES_PATH):
        st.error("Class names file not found.")
        return []
    return np.load(CLASS_NAMES_PATH, allow_pickle=True)

# Initialize app
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="✋",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {max-width: 1000px;}
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background: #f8f9fa;
        text-align: center;
        margin: 1rem 0;
        border-left: 5px solid #4e73df;
    }
    .confidence-bar {
        height: 10px;
        background: #e9ecef;
        border-radius: 5px;
        margin: 10px 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 5px;
        background: #4e73df;
    }
    .history-item {
        display: inline-block;
        margin: 0.3rem;
        padding: 0.5rem;
        border-radius: 0.25rem;
        background: #f1f3f5;
        font-size: 0.9rem;
    }
    .roi-guide {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: rgba(0, 255, 0, 0.5);
        font-size: 1.2rem;
        pointer-events: none;
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.title("✋ Sign Language Recognition System")
st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        Use your webcam to perform sign language gestures. Position your hand in the 
        detection area and hold steady for best results.
    </div>
""", unsafe_allow_html=True)

# Load resources
model = load_model()
class_names = load_class_names()

if model is None or len(class_names) == 0:
    st.stop()

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = deque(maxlen=HISTORY_SIZE)
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold", 0.7, 0.99, MIN_CONFIDENCE, 0.01,
        help="Minimum confidence level to accept a prediction"
    )
    show_confidence = st.checkbox("Show Confidence", True)
    show_history = st.checkbox("Show Prediction History", True)
    enable_smoothing = st.checkbox("Enable Smoothing", True)
    
    st.markdown("---")
    st.subheader("Performance")
    if st.button("Clear History"):
        st.session_state.prediction_history.clear()
    
    st.markdown("---")
    st.subheader("Help Improve")
    with st.form("feedback_form"):
        feedback = st.text_area("Report incorrect predictions or issues")
        if st.form_submit_button("Submit Feedback"):
            if feedback:
                st.session_state.feedback_data.append(feedback)
                st.success("Thank you for your feedback!")

def enhanced_preprocessing(roi):
    """Advanced preprocessing pipeline"""
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # CLAHE for contrast normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(clahe_img, (5,5), 0)
    
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations
    kernel = np.ones((3,3), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Resize and normalize
    resized = cv2.resize(processed, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype('float32') / 255.0
    
    return np.expand_dims(normalized, axis=(0, -1))

def get_smoothed_prediction(predictions):
    """Apply temporal smoothing to predictions"""
    if not enable_smoothing:
        return np.argmax(predictions)
    
    # Add current prediction to history
    current_pred = np.argmax(predictions)
    st.session_state.prediction_history.append(current_pred)
    
    # Return most frequent prediction
    counts = np.bincount(st.session_state.prediction_history)
    return np.argmax(counts)

# Main app layout
col1, col2 = st.columns([2, 1])
with col1:
    start_camera = st.checkbox("🎥 Start Camera", key="camera_active")

frame_placeholder = st.empty()
prediction_placeholder = st.empty()
history_placeholder = st.empty()

if start_camera:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not access webcam. Please check permissions.")
        st.stop()
    
    last_pred_time = time.time()
    current_pred = "None"
    current_conf = 0
    
    while start_camera:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture frame.")
            break

        # Flip and convert frame
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get ROI coordinates
        h, w = frame.shape[:2]
        x1, y1 = (w - ROI_SIZE) // 2, (h - ROI_SIZE) // 2
        x2, y2 = x1 + ROI_SIZE, y1 + ROI_SIZE
        
        # Draw ROI and guide text
        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_rgb, "Place hand here", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        # Process frame every 0.3 seconds
        if time.time() - last_pred_time > 0.3:
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                roi_input = enhanced_preprocessing(roi)
                predictions = model.predict(roi_input, verbose=0)[0]
                
                # Apply smoothing
                pred_idx = get_smoothed_prediction(predictions)
                confidence = predictions[pred_idx]
                
                current_pred = class_names[pred_idx]
                current_conf = confidence
                last_pred_time = time.time()
        
        # Display prediction
        if current_conf > confidence_threshold:
            pred_box = f"""
            <div class='prediction-box'>
                <h2 style='margin: 0;'>Detected: <strong>{current_pred}</strong></h2>
            """
        else:
            pred_box = f"""
            <div class='prediction-box' style='border-left-color: #e74a3b;'>
                <h2 style='margin: 0;'>Detected: <strong style='color: #e74a3b;'>Unknown</strong></h2>
            """
        
        if show_confidence:
            pred_box += f"""
                <div style='margin-top: 0.5rem;'>
                    <div>Confidence: {current_conf:.1%}</div>
                    <div class='confidence-bar'>
                        <div class='confidence-fill' style='width: {current_conf*100:.0f}%'></div>
                    </div>
                </div>
            """
        pred_box += "</div>"
        prediction_placeholder.markdown(pred_box, unsafe_allow_html=True)
        
        # Display prediction history
        if show_history and st.session_state.prediction_history:
            history_html = "<div style='margin-top: 1rem;'>"
            history_html += "<h4>Recent Predictions:</h4><div style='margin-top: 0.5rem;'>"
            
            unique_preds = np.unique(list(st.session_state.prediction_history))
            for pred in unique_preds[-5:]:
                pred_class = class_names[pred]
                count = list(st.session_state.prediction_history).count(pred)
                history_html += f"""
                    <span class='history-item'>
                        {pred_class} <small>({count})</small>
                    </span>
                """
            
            history_html += "</div></div>"
            history_placeholder.markdown(history_html, unsafe_allow_html=True)
        
        # Display frame
        frame_placeholder.image(frame_rgb, channels="RGB")
        
        # Check if camera should stop
        if not st.session_state.camera_active:
            break
    
    cap.release()
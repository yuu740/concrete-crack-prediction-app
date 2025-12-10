import gradio as gr
import joblib
import cv2
import numpy as np
from skimage.feature import hog


model = joblib.load("final_crack_detector_rf.pkl")

def extract_features(image):
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    

    resized = cv2.resize(gray, (224, 224))
    

    features = hog(
        resized, 
        orientations=9, 
        pixels_per_cell=(16, 16),  
        cells_per_block=(2, 2), 
        transform_sqrt=True,       
        block_norm='L1',          
        visualize=False
    )
    
    return features.reshape(1, -1)

def predict_crack(image):
    try:
        if image is None:
            return "Please upload an image."
        
        features = extract_features(image)
        
        prediction = model.predict(features)
        
        if prediction[0] == 1:
            return "⚠️ CRACK DETECTED"        
        else:
            return "✅ NO CRACK DETECTED (Safe)"    
            
    except Exception as e:
        return f"Error: {str(e)}"

interface = gr.Interface(
    fn=predict_crack,
    inputs=gr.Image(label="Upload Concrete Image", sources=["upload", "webcam"]), 
    outputs=gr.Text(label="Analysis Result"),
    title="Concrete Crack Detection (RF + HOG 16x16)",
    description="Random Forest model trained with HOG features (224x224, cell 16x16)."
)

if __name__ == "__main__":
    interface.launch()
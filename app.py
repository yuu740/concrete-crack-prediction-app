import gradio as gr
import joblib
import cv2
import numpy as np
from skimage.feature import hog
import time

try:
    model = joblib.load("final_crack_detector_rf.pkl")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

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

def analyze_crack(image):
    if image is None:
        return "Please upload an image.", "0.00%", "0.0000 s"
    
    if model is None:
        return "Model not loaded.", "0.00%", "0.0000 s"

    start_time = time.perf_counter()
    
    try:
        features = extract_features(image)
        probabilities = model.predict_proba(features)[0]
        prediction = np.argmax(probabilities)
        confidence = probabilities[prediction] * 100
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        if prediction == 1:
            res_text = "CRACK DETECTED"
        else:
            res_text = "NO CRACK DETECTED"
            
        return res_text, f"{confidence:.2f}%", f"{elapsed_time:.4f} s"

    except Exception as e:
        return f"Error: {str(e)}", "0.00%", "0.0000 s"


with gr.Blocks() as demo: 
    
    gr.Markdown("# Concrete Crack Detection")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Upload Image", sources=["upload", "webcam"], type="numpy")
            analyze_btn = gr.Button("Analyze", variant="primary")
        
        with gr.Column():
            res_out = gr.Textbox(label="Analysis Result", interactive=False)
            
            with gr.Row():
                conf_out = gr.Textbox(label="Confidence Score")
                time_out = gr.Textbox(label="Processing Time")

    analyze_btn.click(analyze_crack, inputs=input_img, outputs=[res_out, conf_out, time_out])

if __name__ == "__main__":
    demo.launch()
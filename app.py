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
        return (
            "<div class='status-card neutral'><div class='text-content'>Upload an Image</div><div class='subtitle'>Drop your concrete image here</div></div>", 
            "<div class='metric-card'><div class='metric-value'>-</div></div>", 
            "<div class='metric-card'><div class='metric-value'>-</div></div>"
        )
    
    if model is None:
        return (
            "<div class='status-card error'><div class='text-content'>Model Error</div></div>", 
            "<div class='metric-card'><div class='metric-value'>-</div></div>", 
            "<div class='metric-card'><div class='metric-value'>-</div></div>"
        )

    start_time = time.perf_counter()
    
    try:
        features = extract_features(image)
        probabilities = model.predict_proba(features)[0]
        prediction = np.argmax(probabilities)
        confidence = probabilities[prediction] * 100
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        if prediction == 1:
            res_html = f"""
            <div class='status-card danger'>
                <div class='text-content'>Crack Detected</div>
                <div class='subtitle'>Immediate inspection recommended</div>
            </div>
            """
        else:
            res_html = f"""
            <div class='status-card safe'>
                <div class='text-content'>No Crack Detected</div>
                <div class='subtitle'>Structure appears safe</div>
            </div>
            """
        conf_html = f"<div class='metric-card'><div class='metric-value'>{confidence:.1f}%</div></div>"
        time_html = f"<div class='metric-card'><div class='metric-value'>{elapsed_time*1000:.1f}ms</div></div>"
            
        return res_html, conf_html, time_html
    except Exception as e:
        return (
            f"<div class='status-card error'><div class='text-content'>Error</div><div class='subtitle'>{str(e)}</div></div>", 
            "<div class='metric-card'><div class='metric-value'>-</div></div>", 
            "<div class='metric-card'><div class='metric-value'>-</div></div>"
        )

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
}

body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}

#component-0 {
    background: rgba(255, 255, 255, 0.98) !important;
    backdrop-filter: blur(20px) !important;
    border-radius: 24px !important;
    padding: 40px !important;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25) !important;
}

.title-container {
    text-align: center;
    margin-bottom: 40px;
    padding: 0;
    background: transparent;
    border: none;
}

.title-container h1 {
    font-size: 2.5em !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 !important;
    letter-spacing: -0.5px;
}

.title-container p {
    color: #64748b;
    font-size: 1.1em;
    margin-top: 12px;
    font-weight: 500;
}

.status-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px 30px;
    border-radius: 20px;
    font-weight: 600;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    min-height: 200px;
    position: relative;
    overflow: hidden;
}

.status-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, transparent, currentColor, transparent);
    opacity: 0.6;
}

.icon {
    font-size: 64px;
    margin-bottom: 16px;
    animation: fadeInScale 0.5s ease;
}

.text-content {
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 8px;
}

.subtitle {
    font-size: 15px;
    opacity: 0.75;
    font-weight: 500;
    margin-top: 8px;
}

.neutral { 
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    color: #64748b; 
    border: 2px dashed #cbd5e1; 
}

.safe { 
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    color: #047857; 
    border: 2px solid #10b981; 
}

.danger { 
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    color: #dc2626; 
    border: 2px solid #ef4444; 
    animation: pulseGlow 2s infinite;
}

.error { 
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    color: #991b1b; 
    border: 2px solid #dc2626;
}

.metric-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 2px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
    transition: all 0.3s ease;
    height: 100%;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
}

.metric-value {
    font-size: 32px;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.metric-label {
    font-size: 13px;
    text-transform: uppercase;
    color: #64748b;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 12px;
    text-align: center;
    display: block;
}

button.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 18px 36px !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4) !important;
    transition: all 0.3s ease !important;
    letter-spacing: 0.5px;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 32px rgba(102, 126, 234, 0.5) !important;
}

.image-container {
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 2px solid #e2e8f0 !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04) !important;
}

h3 {
    font-size: 1.4em !important;
    font-weight: 700 !important;
    color: #1e293b !important;
    margin-bottom: 20px !important;
    letter-spacing: -0.5px;
}

@keyframes pulseGlow {
    0%, 100% { 
        box-shadow: 0 8px 32px rgba(239, 68, 68, 0.2);
    }
    50% { 
        box-shadow: 0 8px 32px rgba(239, 68, 68, 0.4), 0 0 0 8px rgba(239, 68, 68, 0.1);
    }
}

@keyframes fadeInScale {
    from {
        opacity: 0;
        transform: scale(0.8);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}
"""

with gr.Blocks(css=custom_css, title="Crack Detector", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown(
        """
        <div class="title-container">
            <h1>Crack Detection System</h1>
        </div>
        """
    )
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            input_img = gr.Image(
                label="Upload Image", 
                sources=["upload", "webcam", "clipboard"], 
                type="numpy",
                height=400,
                elem_classes="image-container"
            )
            analyze_btn = gr.Button("Analyze Image", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("Analysis Results")
            res_out = gr.HTML(
                value="<div class='status-card neutral'><div class='text-content'>Ready to Analyze</div><div class='subtitle'>Upload an image to get started</div></div>"
            )
            
            with gr.Row():
                with gr.Column():
                    conf_out = gr.HTML(value="<div class='metric-card'><div class='metric-value'>-</div></div>")
                
                with gr.Column():
                    time_out = gr.HTML(value="<div class='metric-card'><div class='metric-value'>-</div></div>")

    analyze_btn.click(
        analyze_crack, 
        inputs=input_img, 
        outputs=[res_out, conf_out, time_out]
    )

if __name__ == "__main__":
    demo.launch()
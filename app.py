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
            </div>
            """
        else:
            res_html = f"""
            <div class='status-card safe'>
                <div class='text-content'>No Crack Detected</div>
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

body, .gradio-container {
    background-color: #0f172a !important; 
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%) !important;
    color: #e2e8f0 !important; 
}

.gradio-container {
    max-width: 98% !important; 
    width: 98% !important;
    margin: auto !important;
    padding-left: 20px !important;
    padding-right: 20px !important;
}

#component-0 {
    background: rgba(30, 41, 59, 0.7) !important; 
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 24px !important;
    padding: 30px !important;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5) !important;
}

.two-col {
    display: flex !important;
    flex-direction: row !important;
    gap: 30px !important;
    align-items: flex-start !important;
}

@media (max-width: 800px) {
    .two-col {
        flex-direction: column !important;
    }
}

.title-container h1 {
    font-size: 2.5em !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #a5b4fc 0%, #c084fc 100%); 
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 !important;
}

span, label, p {
    color: #cbd5e1 !important;
}

.status-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px 30px;
    border-radius: 20px;
    font-weight: 600;
    min-height: 200px;
    height: 100%;
    position: relative;
    border: 1px solid rgba(255,255,255,0.1);
}

.text-content {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 8px;
    color: #ffffff;
}

.subtitle {
    font-size: 15px;
    opacity: 0.8;
    font-weight: 500;
    color: #e2e8f0;
}

.neutral { 
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 2px dashed #475569; 
}

.safe { 
    background: linear-gradient(135deg, #064e3b 0%, #065f46 100%); 
    border: 2px solid #10b981; 
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.2);
}
.safe .text-content { color: #34d399; }

.danger { 
    background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%); 
    border: 2px solid #ef4444; 
    animation: pulseGlow 2s infinite;
}
.danger .text-content { color: #fca5a5; }

.error { 
    background: #450a0a;
    border: 2px solid #dc2626;
}

.metric-card {
    background: #1e293b !important; 
    border: 1px solid #334155 !important;
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    height: 100%;
}

.metric-value {
    font-size: 32px;
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc 0%, #c084fc 100%); 
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

button.primary {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    padding: 15px !important;
    margin-top: 20px !important;
}
button.primary:hover {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    box-shadow: 0 0 15px rgba(139, 92, 246, 0.4) !important;
}

.image-container {
    border: 1px solid #475569 !important;
    background-color: #1e293b !important;
    border-radius: 16px !important;
}

h3 {
    color: #e2e8f0 !important; 
    font-size: 1.4em !important;
    margin-bottom: 15px !important;
}

@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 15px rgba(239, 68, 68, 0.2); }
    50% { box-shadow: 0 0 25px rgba(239, 68, 68, 0.5); }
}
"""


with gr.Blocks(css=custom_css, title="Crack Detector", theme=gr.themes.Default()) as demo:
    
    gr.Markdown(
        """
        <div class="title-container">
            <h1>Crack Detection System</h1>
        </div>
        """
    )

    with gr.Row(elem_classes="two-col"):

        with gr.Column(scale=1): 
            gr.Markdown("### Input Image")
            input_img = gr.Image(
                label=None,
                sources=["upload", "webcam", "clipboard"],
                type="numpy",
                height=450,
                elem_classes="image-container"
            )
            analyze_btn = gr.Button("Analyze Structure", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### Analysis Results")

            res_out = gr.HTML(
                value="""
                <div class='status-card neutral'>
                    <div class='text-content'>Ready</div>
                    <div class='subtitle'>System Awaiting Input</div>
                </div>
                """
            )
            
            with gr.Row(elem_id="metrics-row"):
                with gr.Column(min_width=100):
                    gr.Markdown("<div style='text-align:center; color:#94a3b8; font-weight:700; margin-bottom:5px'>CONFIDENCE</div>")
                    conf_out = gr.HTML("<div class='metric-card'><div class='metric-value'>-</div></div>")
                with gr.Column(min_width=100):
                    gr.Markdown("<div style='text-align:center; color:#94a3b8; font-weight:700; margin-bottom:5px'>LATENCY</div>")
                    time_out = gr.HTML("<div class='metric-card'><div class='metric-value'>-</div></div>")

    analyze_btn.click(
        analyze_crack, 
        inputs=input_img, 
        outputs=[res_out, conf_out, time_out]
    )

if __name__ == "__main__":
    demo.launch()
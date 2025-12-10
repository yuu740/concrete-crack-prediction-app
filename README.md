---
title: Concrete Crack Detection (RF + HOG 16x16)
emoji: 🧱
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 4.10.0 
app_file: app.py
pinned: false
---

# Concrete Crack Detection Application

This space hosts a simple image classification application to detect cracks in concrete surfaces. 
It uses the Random Forest algorithm trained on HOG (Histogram of Oriented Gradients) features.

**Configuration:**
- **Model:** Random Forest (`final_crack_detector_rf.pkl`)
- **Feature:** HOG (224x224, 16x16 cell size)
- **SDK:** Gradio
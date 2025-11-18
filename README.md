# Vehicle Damage Detection
A deep learning application that automatically detects and classifies vehicle damage from images. Simply drag and drop a car image, and the model identifies the type of damage present.

## Overview
This application uses a convolutional neural network to analyze car damage from quarter-panel views. The model specifically works with third-quarter front or rear view images to accurately classify damage types.

<img width="615" height="701" alt="Front Breakage" src="https://github.com/user-attachments/assets/5738f6ab-c4d1-4bf9-bba7-fbf189cdfab3" />


Architecture: ResNet50 with transfer learning

Training data: ~1,700 images across 6 damage categories

Validation accuracy: 80%

### Damage Classes
1. Front Normal

2. Front Crushed

3. Front Breakage

4. Rear Normal

5. Rear Crushed

6. Rear Breakage

Hyperparameter Optimization
Optuna was used for automated hyperparameter tuning with 20 trials, resulting in:

Learning rate: 0.0019

Dropout: 0.44

This optimization improved model stability and maintained consistent validation performance.
Setup
Install dependencies:

bash
pip install -r requirements.txt

Run the Streamlit app:

bash
streamlit run app.py

Tech Stack
Python

PyTorch

Streamlit

ResNet50 (pretrained)

Optuna (hyperparameter optimization)

Usage
Upload a third-quarter front or rear view image of a vehicle. The model will analyze and classify the damage type.

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import streamlit as st

# Replace with your real model architecture
class VehicleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3*224*224, 3)  # 3 classes example

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Load model once and cache it
@st.cache_resource
def load_model(model_path="model/saved_model.pth"):
    model = VehicleModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

model = load_model()

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    classes = ["minor_damage", "moderate_damage", "severe_damage"]
    return classes[predicted_class]

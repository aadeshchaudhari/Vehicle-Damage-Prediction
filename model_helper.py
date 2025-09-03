import torch
import torch.nn as nn
from torchvision import models
import os
from PIL import Image
from torchvision import transforms

# Define your model architecture (must match training)
class VehicleModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = models.resnet34(pretrained=False)  # Must match your training
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Define absolute path for the model (works in Streamlit Cloud)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "saved_model.pth")

# Load model once
def load_model(model_path=MODEL_PATH):
    model = VehicleModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

# Load the model
model = load_model()

# Preprocessing and prediction
def predict(image_path):
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Forward pass
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_class = torch.argmax(outputs, dim=1).item()

    # Map to labels
    classes = ["minor_damage", "moderate_damage", "severe_damage"]
    return classes[predicted_class]

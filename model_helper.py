import os
import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

# Relative path to model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "saved_model.pth")

# Define model architecture in case checkpoint is a state_dict
class VehicleModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = models.resnet34(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Robust loader
def load_model(model_path=MODEL_PATH):
    checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, nn.Module):
        # Full model saved
        model = checkpoint
    elif isinstance(checkpoint, dict):
        # Check for common keys
        if "model_state_dict" in checkpoint:
            model = VehicleModel()
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Assume raw state_dict
            model = VehicleModel()
            model.load_state_dict(checkpoint)
    else:
        raise ValueError("Unsupported checkpoint type")

    model.eval()
    return model

# Load model once
model = load_model()

# Prediction function
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0]()

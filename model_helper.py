import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Relative path for the model inside your repo
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "saved_model.pth")

# Define model architecture
class VehicleModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = models.resnet34(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Robust model loading
def load_model(model_path=MODEL_PATH):
    checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, nn.Module):
        # full model
        model = checkpoint
    elif isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model = VehicleModel()
            model.load_state_dict(checkpoint["model_state_d]()

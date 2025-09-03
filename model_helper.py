import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Relative path to model folder
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "saved_model.pth")

class VehicleModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = models.resnet34(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def load_model(model_path=MODEL_PATH):
    checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model = VehicleModel()
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model = VehicleModel()
            model.load_state_dict(checkpoint)
    else:
        raise ValueError("Unsupported checkpoint format")

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
        outputs = model(image_tensor)
        predicted_class = torch.argmax(outputs, dim=1).item()
    classes = ["minor_damage", "moderate_damage", "severe_damage"]
    return classes[predicted_class]

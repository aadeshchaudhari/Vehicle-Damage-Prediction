import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms

# Path to your saved model checkpoint
MODEL_PATH = "saved_model.pth"

# Define your model architecture (adjust to match your checkpoint as closely as possible)
class VehicleModel(nn.Module):
    def __init__(self, num_classes=3):  # replace num_classes with your output classes
        super(VehicleModel, self).__init__()
        self.model = models.resnet50(weights=None)  # no pretrained weights
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# Load model on CPU
def load_model(model_path=MODEL_PATH):
    model = VehicleModel()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)  # ignore missing/unexpected keys
    model.eval()
    return model

# Initialize the model
model = load_model()

# Prediction function
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

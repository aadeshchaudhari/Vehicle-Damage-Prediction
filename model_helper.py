import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Replace with your actual model definition
class VehicleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Example: simple linear layer (replace with your real model)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3*224*224, 3)  # Example: 3 classes

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Load model on CPU
def load_model(model_path="model/saved_model.pth"):
    model = VehicleModel()
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

# Preprocessing and prediction
def predict(image_path):
    model = load_model()

    # Define preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),   # Resize to model input
        transforms.ToTensor(),           # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load image
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Forward pass
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_class = torch.argmax(outputs, dim=1).item()

    # Map class index to label (replace with your real classes)
    classes = ["minor_damage", "moderate_damage", "severe_damage"]
    return classes[predicted_class]

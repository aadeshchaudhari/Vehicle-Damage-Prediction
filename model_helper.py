import os
import torch
from PIL import Image
from torchvision import transforms

# Relative path to model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "saved_model.pth")

# Load full model directly
def load_model(model_path=MODEL_PATH):
    model = torch.load(model_path, map_location="cpu")
    model.eval()
    return model

# Load once at startup
model = load_model()

# Prediction function
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

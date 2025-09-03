import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Example model class (replace this with your actual model structure)
class VehicleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Dummy example: 4 output classes
        self.fc = nn.Linear(3 * 224 * 224, 4)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        return self.fc(x)

# Load model
def load_model(model_path="model/saved_model.pth"):
    model = VehicleModel()
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

# Predict function
def predict(image_path):
    model = load_model()

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # add batch dimension

    # Run prediction
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    # Map numeric class to human-readable labels
    class_labels = {0: "No Damage", 1: "Minor Damage", 2: "Moderate Damage", 3: "Severe Damage"}
    return class_labels.get(predicted_class, "Unknown")


import torch
import torch.nn as nn
from torchvision import models

# Load pretrained model architecture (replace with what you trained)
class VehicleModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = models.resnet34(pretrained=False)  # must match training
        # Replace the final fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load model
def load_model(model_path="model/saved_model.pth"):
    model = VehicleModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

model = load_model()

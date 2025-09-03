import torch
import torch.nn as nn

# Example: replace with your actual model class
class VehicleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # your layers here
    def forward(self, x):
        return x  # dummy

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
    # load and preprocess image
    # convert to tensor
    # call model(image_tensor)
    # return predicted class
    return "dummy_class"  # placeholder until real preprocessing is added

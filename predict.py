from cog import BasePredictor, Input, Path
import torch

class Predictor(BasePredictor):
    def setup(self):
        """Loads the model into memory to make running multiple predictions efficient"""
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        self.model.eval()

    def predict(self, image: Path = Input(description="Image to classify")) -> str:
        """Runs a single prediction on the model"""
        # ... implement your prediction logic here ...
        return "Not implemented yet"

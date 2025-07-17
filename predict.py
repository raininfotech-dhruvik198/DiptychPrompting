from cog import BasePredictor, Input, Path
import torch
from torchvision import transforms
from PIL import Image
import json

class Predictor(BasePredictor):
    def setup(self):
        """Loads the model into memory to make running multiple predictions efficient"""
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # Download ImageNet labels
        self.labels = torch.hub.load_state_dict_from_url("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")


    def predict(self, image: Path = Input(description="Image to classify")) -> str:
        """Runs a single prediction on the model"""
        img = Image.open(image).convert("RGB")
        input_tensor = self.preprocess(img)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')

        with torch.no_grad():
            output = self.model(input_batch)

        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        _, index = torch.max(output, 1)

        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        return self.labels[index[0]]

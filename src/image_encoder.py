import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from PIL import Image


# Image Encoder using Swin Transformer V2 Large model from Microsoft
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-large-patch4-window12-192-22k")
        self.model = AutoModel.from_pretrained("microsoft/swinv2-large-patch4-window12-192-22k")
        
        # Freeze the Swin Transformer model
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images):
        # Process images through the pre-trained model
        inputs = self.processor(images=images, return_tensors="pt")
        outputs = self.model(**inputs)
        # Extract and return image embeddings
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings

"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# Image Encoder using Swin Transformer V2 Large model from Microsoft
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        # Initialize the CLIP processor and model
        self.processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
        self.model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
        
        # Freeze the CLIP model parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images):
        # Preprocess the images with the CLIP processor
        inputs = self.processor(images=images, return_tensors="pt")
        
        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        for key in inputs:
            inputs[key] = inputs[key].to(device)

        # Pass inputs through the CLIP model
        outputs = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        
        # Normalize embeddings (common practice with CLIP embeddings)
        embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
        
        return embeddings
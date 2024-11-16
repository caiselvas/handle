import torch
import torch.nn as nn
from transformers import ViTModel

# Image Encoder using Vision Transformer
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        # Load a pre-trained Vision Transformer
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    def forward(self, x):
        # Extract the [CLS] token as the image representation
        return self.vit(x).last_hidden_state[:, 0]  # Shape: [batch_size, hidden_dim]
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from PIL import Image



class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-large-patch4-window12-192-22k")
        self.model = AutoModel.from_pretrained("microsoft/swinv2-large-patch4-window12-192-22k")

    def forward(self, images):
        # Procesar las imágenes
        inputs = self.processor(images=images, return_tensors="pt")
        # Obtener las salidas del modelo
        outputs = self.model(**inputs)
        # Extraer las representaciones de las imágenes
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings

"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
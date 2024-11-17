import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel

# Image Encoder using Fashion CLIP model
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        
        # Cargar el procesador y el modelo CLIP específicos para FashionCLIP
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        
        # Congelar los parámetros del modelo CLIP
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images):
        # Procesar las imágenes usando el procesador CLIP
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=False)
        
        # Mover inputs al dispositivo adecuado
        device = next(self.model.parameters()).device
        for key in inputs:
            inputs[key] = inputs[key].to(device)

        # Obtener características de la imagen usando el modelo CLIP
        embeddings = self.model.get_image_features(**inputs)
        return embeddings

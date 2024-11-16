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

# Tabular Encoder for categorical data
class TabularEncoder(nn.Module):
    def __init__(self, num_categories, embedding_dim):
        super(TabularEncoder, self).__init__()
        # Create an embedding layer for each categorical feature
        self.embeddings = nn.ModuleList([nn.Embedding(cat_size, embedding_dim) for cat_size in num_categories])
        # Transformer layer to capture inter-column dependencies
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4)

    def forward(self, x):
        # Generate embeddings for each categorical feature
        embedded_columns = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        # Stack all embeddings to form a sequence of tokens (each representing a column)
        x = torch.stack(embedded_columns, dim=1)  # Shape: [batch_size, num_columns, embedding_dim]
        # Apply Transformer layer for inter-column attention
        return self.transformer_layer(x)  # Shape: [batch_size, num_columns, embedding_dim]

# Multimodal Interaction Module with Cross Attention
class MultimodalInteraction(nn.Module):
    def __init__(self, embedding_dim, num_layers):
        super(MultimodalInteraction, self).__init__()
        # Define a stack of transformer layers to perform cross-attention between image and tabular features
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4) for _ in range(num_layers)])

    def forward(self, image_features, tabular_features):
        # Concatenate image and tabular features
        combined = torch.cat([image_features.unsqueeze(1), tabular_features], dim=1)  # Shape: [batch_size, 1 + num_columns, embedding_dim]
        # Pass through each transformer layer to allow cross-attention
        for layer in self.layers:
            combined = layer(combined)
        return combined[:, 0, :]  # Return the [CLS] token representation for classification

# TIP Model integrating ImageEncoder, TabularEncoder, and Multimodal Interaction Module
class TIPModel(nn.Module):
    def __init__(self, num_categories, embedding_dim, num_classes):
        super(TIPModel, self).__init__()
        self.image_encoder = ImageEncoder()
        self.tabular_encoder = TabularEncoder(num_categories, embedding_dim)
        self.multimodal_interaction = MultimodalInteraction(embedding_dim, num_layers=2)
        # Final classification layer
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, img, tab):
        # Encode image and tabular inputs
        img_features = self.image_encoder(img)  # Shape: [batch_size, embedding_dim]
        tab_features = self.tabular_encoder(tab)  # Shape: [batch_size, num_columns, embedding_dim]
        # Perform cross-attention between image and tabular features
        fused_features = self.multimodal_interaction(img_features, tab_features)  # Shape: [batch_size, embedding_dim]
        # Classification
        return self.classifier(fused_features)  # Shape: [batch_size, num_classes]

# Example instantiation
num_categories = [10, 15, 20]  # Number of categories for each categorical feature
embedding_dim = 64
num_classes = 5
model = TIPModel(num_categories, embedding_dim, num_classes)
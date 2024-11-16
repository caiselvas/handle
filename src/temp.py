import torch
import torch.nn as nn
from transformers import ViTModel

# Image Encoder using Vision Transformer (ViT)
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    def forward(self, x):
        # Extract image representation using the [CLS] token
        return self.vit(x).last_hidden_state[:, 0]  # Shape: [batch_size, hidden_dim]

# Tabular Encoder for categorical data
class TabularEncoder(nn.Module):
    def __init__(self, num_categories, embedding_dim):
        super(TabularEncoder, self).__init__()
        # Embedding for each categorical feature
        self.embeddings = nn.ModuleList([nn.Embedding(cat_size, embedding_dim) for cat_size in num_categories])
        # Column embeddings to encode column identity
        self.column_embeddings = nn.Parameter(torch.randn(len(num_categories), embedding_dim))
        # Transformer layer for capturing inter-column dependencies
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4)

    def forward(self, x):
        # Embed each column and add column-specific embeddings
        embedded_columns = [emb(x[:, i]) + self.column_embeddings[i] for i, emb in enumerate(self.embeddings)]
        x = torch.stack(embedded_columns, dim=1)  # Shape: [batch_size, num_columns, embedding_dim]
        # Pass through transformer for inter-column attention
        return self.transformer_layer(x)  # Shape: [batch_size, num_columns, embedding_dim]

# Multimodal Interaction Module with Residual Connections and Layer Normalization
class MultimodalInteraction(nn.Module):
    def __init__(self, embedding_dim, num_layers):
        super(MultimodalInteraction, self).__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4) for _ in range(num_layers)])
        self.cross_attention = nn.MultiheadAttention(embedding_dim, num_heads=4)
        self.layer_norm = nn.LayerNorm(embedding_dim)  # Layer normalization for residual connections
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )

    def forward(self, image_features, tabular_features):
        # Expand image_features for cross attention
        img_expanded = image_features.unsqueeze(1).expand(-1, tabular_features.size(1), -1)
        
        # Concatenate image and tabular embeddings
        combined = torch.cat([img_expanded, tabular_features], dim=1)  # Shape: [batch_size, num_columns + 1, embedding_dim]

        for layer in self.layers:
            # Self-attention with residual connection and normalization
            combined_res = combined
            combined = self.layer_norm(combined + layer(combined))

            # Cross-attention with residual connection and normalization
            cross_attn_res = combined
            combined, _ = self.cross_attention(combined, combined, combined)
            combined = self.layer_norm(cross_attn_res + combined)

            # Feed-forward with residual connection and normalization
            ff_res = combined
            combined = self.layer_norm(ff_res + self.feed_forward(combined))

        return combined[:, 0, :]  # Return the [CLS] token representation for classification

# TIP Model integrating ImageEncoder, TabularEncoder, and Multimodal Interaction Module
class TIPModel(nn.Module):
    def __init__(self, num_categories, embedding_dim, num_classes):
        super(TIPModel, self).__init__()
        self.image_encoder = ImageEncoder()
        self.tabular_encoder = TabularEncoder(num_categories, embedding_dim)
        self.multimodal_interaction = MultimodalInteraction(embedding_dim, num_layers=2)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, img, tab):
        # Encode image and tabular inputs
        img_features = self.image_encoder(img)  # Shape: [batch_size, embedding_dim]
        tab_features = self.tabular_encoder(tab)  # Shape: [batch_size, num_columns, embedding_dim]
        
        # Perform cross-attention between image and tabular features
        fused_features = self.multimodal_interaction(img_features, tab_features)  # Cross-attention fusion
        
        # Classification
        return self.classifier(fused_features)  # Shape: [batch_size, num_classes]

# Example usage
num_categories = [10, 15, 20]  # Number of categories per categorical feature
embedding_dim = 64
num_classes = 5
model = TIPModel(num_categories, embedding_dim, num_classes)
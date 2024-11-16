import torch
import torch.nn as nn

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
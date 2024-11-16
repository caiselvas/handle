import torch
import torch.nn as nn

# Multimodal Interaction Module with Image as Queries
class MultimodalInteraction(nn.Module):
    def __init__(self, embedding_dim, num_layers):
        super(MultimodalInteraction, self).__init__()
        # Define cross attention with image as queries
        self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )

    def forward(self, image_features, tabular_features):
        # Image features as Queries, tabular features as Keys and Values
        image_features = image_features.unsqueeze(1)  # Shape: [batch_size, 1, embedding_dim]
        image_features = image_features.transpose(0, 1)  # Required shape: [seq_len, batch_size, embedding_dim]
        tabular_features = tabular_features.transpose(0, 1)  # Shape: [num_columns, batch_size, embedding_dim]
        
        # Cross-attention with image as queries and tabular data as keys/values
        attn_output, _ = self.cross_attention(query=image_features, key=tabular_features, value=tabular_features)
        attn_output = attn_output.transpose(0, 1).squeeze(1)  # Shape back to [batch_size, embedding_dim]
        
        # Residual connection and layer norm
        attn_output = self.layer_norm(image_features.squeeze(1) + attn_output)
        
        # Feed-forward layer with residual connection
        ff_output = self.feed_forward(attn_output)
        combined_output = self.layer_norm(attn_output + ff_output)
        
        return combined_output  # Combined output for final classification
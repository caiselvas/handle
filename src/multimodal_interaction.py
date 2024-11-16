import torch
import torch.nn as nn

# Multimodal Interaction Module with Image as Queries
class MultimodalInteraction(nn.Module):
    """
    Multimodal Interaction Module with multiple Transformer blocks using cross-attention.
    
    This module integrates image and tabular data, using the image as the primary source of information (queries)
    and tabular data as supporting context (keys and values). It consists of a series of cross-attention and 
    feed-forward layers, each with residual connections and layer normalization, which allows the image features 
    to iteratively refine their representation based on the tabular features.
    
    Args:
        embedding_dim (int): The dimensionality of the embeddings for both image and tabular features.
        num_blocks (int): The number of Transformer blocks (cross-attention + feed-forward) in the module.
    """
    
    def __init__(self, embedding_dim: int, num_blocks: int):
        super(MultimodalInteraction, self).__init__()
        # Define multiple cross-attention layers, one for each layer in num_layers
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4) for _ in range(num_blocks)
        ])
        
        # Define a feed-forward layer for each Transformer block
        self.feed_forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim * 4),
                nn.ReLU(),
                nn.Linear(embedding_dim * 4, embedding_dim)
            ) for _ in range(num_blocks)
        ])
        
        # Layer normalization for residual connections in attention and feed-forward layers
        self.layer_norms_attn = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_blocks)])
        self.layer_norms_ff = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_blocks)])

    def forward(self, image_features, tabular_features):
        """
        Forward pass for the Multimodal Interaction Module.
        
        Args:
            image_features (Tensor): The image embeddings, with shape [batch_size, embedding_dim].
            tabular_features (Tensor): The tabular embeddings, with shape [batch_size, num_columns, embedding_dim].
        
        Returns:
            Tensor: Refined image features, with shape [batch_size, embedding_dim].
        """
        
        # Prepare image features as Queries (Q) and expand for cross-attention
        image_features = image_features.unsqueeze(1).transpose(0, 1)  # Shape: [1, batch_size, embedding_dim]
        
        # Prepare tabular features as Keys (K) and Values (V) for cross-attention
        tabular_features = tabular_features.transpose(0, 1)  # Shape: [num_columns, batch_size, embedding_dim]

        # Apply each Transformer block (cross-attention + feed-forward)
        for cross_attention, feed_forward, norm_attn, norm_ff in zip(
                self.cross_attention_layers, self.feed_forward_layers, self.layer_norms_attn, self.layer_norms_ff):
            
            # Cross-attention layer: Image as Queries, Tabular as Keys and Values
            attn_output, _ = cross_attention(query=image_features, key=tabular_features, value=tabular_features)
            
            # Residual connection and layer normalization after cross-attention
            attn_output = norm_attn(image_features + attn_output)

            # Feed-forward layer with residual connection and normalization
            ff_output = feed_forward(attn_output)
            image_features = norm_ff(attn_output + ff_output)  # Update image_features after each layer

        # Return the final refined image features
        return image_features.squeeze(0)  # Shape: [batch_size, embedding_dim]
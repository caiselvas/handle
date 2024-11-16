from typing import List

import torch
import torch.nn as nn

class TabularEncoder(nn.Module):
    def __init__(self, num_categories_list: List[int], embedding_dim: int, num_blocks: int):
        """
        Tabular Encoder for categorical data using embeddings and Transformer layers.

        Args:
            num_categories_list (List[int]): A list of integers representing the number of categories for each categorical feature.
            embedding_dim (int): The dimension of the embeddings for each categorical feature.
            num_blocks (int): The number of Transformer blocks (self-attention + feed-forward) in the encoder.
        """
        super(TabularEncoder, self).__init__()

        # Embedding layers for each categorical feature (column)
        self.embeddings = nn.ModuleList([nn.Embedding(cat_size, embedding_dim) for cat_size in num_categories_list])
        
        # Column-specific embeddings to encode the identity of each column
        self.column_embeddings = nn.Parameter(torch.randn(len(num_categories_list), embedding_dim))
        
        # Multiple Transformer blocks for inter-column attention
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4)
            for _ in range(num_blocks)
        ])
        
        # Layer normalization for each Transformer block
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_blocks)])

    def forward(self, x):
        # Generate embeddings for each categorical feature and add column-specific embeddings
        embedded_columns = [emb(x[:, i]) + self.column_embeddings[i] for i, emb in enumerate(self.embeddings)]
        
        # Stack embeddings to form a sequence of tokens (one token per column)
        x = torch.stack(embedded_columns, dim=1)  # Shape: [batch_size, num_columns, embedding_dim]
        
        # Apply each Transformer layer with residual connections and normalization
        for transformer_layer, layer_norm in zip(self.transformer_layers, self.layer_norms):
            # Residual connection with layer normalization
            x = layer_norm(x + transformer_layer(x))
        
        return x  # Final shape: [batch_size, num_columns, embedding_dim]
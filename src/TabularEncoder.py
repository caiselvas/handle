import torch
import torch.nn as nn

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
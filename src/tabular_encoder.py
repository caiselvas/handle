import torch
import torch.nn as nn

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
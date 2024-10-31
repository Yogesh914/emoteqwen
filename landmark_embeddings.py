import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
import math

# Adapter module to map facial landmarks and blendshapes to token embeddings
class MLPEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(MLPEmbedding, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim*4, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(input_dim*4, input_dim*2, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(input_dim*2, input_dim, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(input_dim, embedding_dim, dtype=torch.bfloat16)
        )



    def forward(self, x):
        # x shape: (batch_size, num_seconds, n_landmarks, 3)
        batch_size = x.shape[0]
        num_seconds = x.shape[1]
        x = x.view(batch_size, num_seconds, -1)
        embeddings = self.mlp(x)
        return embeddings  # shape: (batch_size, num_seconds, embedding_dim)



class TransformerEmbedding(nn.Module):
    def __init__(self, embedding_dim, num_landmarks=478, input_dim=3, n_layers=4, n_heads=8):
        super(TransformerEmbedding, self).__init__()

        self.input_proj = nn.Linear(input_dim, 1024, dtype=torch.bfloat16)

        self.landmark_pos_encoding = nn.Parameter(
            torch.randn(1, num_landmarks, 1024, dtype=torch.bfloat16)
        )

        self.spatial_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=1024, 
                nhead=n_heads, 
                activation='gelu', 
                batch_first=True, 
                dtype=torch.bfloat16
            ),
            num_layers=n_layers
        )

        self.proj = nn.Linear(num_landmarks * 1024, embedding_dim, dtype=torch.bfloat16)

    def forward(self, x):
        # x shape: (batch_size, num_seconds, n_landmarks, 3)
        b, t, l, c = x.size()
        x = x.view(b*t, l, c) # shape: (b*30, 478, 3)
        x = self.input_proj(x) # shape: (b*30, 478, d)
        x = x + self.landmark_pos_encoding

        x = self.spatial_transformer(x) 
        x = x.view(b, t, -1) # shape: (b, 30, 478*d)
        x = self.proj(x) # shape: (b, 30, d)

        return x

def get_embedding_model(embedding_type, **kwargs):
    if embedding_type == 'mlp':
        return MLPEmbedding(**kwargs)
    elif embedding_type == 'transformer':
        return TransformerEmbedding(**kwargs)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
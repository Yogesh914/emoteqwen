import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

# Adapter module to map facial landmarks and blendshapes to token embeddings
class MLPEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(MLPEmbedding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4096, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(4096, 3072, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(3072, embedding_dim, dtype=torch.bfloat16)
        )

    def forward(self, x):
        # x shape: (batch_size, num_seconds, n_landmarks, 3)
        batch_size = x.shape[0]
        num_seconds = x.shape[1]
        x = x.view(batch_size, num_seconds, -1)
        embeddings = self.mlp(x)
        return embeddings  # shape: (batch_size, num_seconds, embedding_dim)



class PointNetEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(PointNetEmbedding, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, 1, dtype=torch.bfloat16)
        self.conv2 = nn.Conv1d(64, 128, 1, dtype=torch.bfloat16)
        self.conv3 = nn.Conv1d(128, 256, 1, dtype=torch.bfloat16)
        self.fc = nn.Linear(256, embedding_dim, dtype=torch.bfloat16)
        self.gelu = nn.GELU()



    def forward(self, x):

        batch_size = x.shape[0]
        num_seconds = x.shape[1]
        n_landmarks = x.shape[2]

        x = x.view(batch_size * num_seconds, n_landmarks, 3)
        x = x.transpose(1,2)

        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        x = self.gelu(self.conv3(x))

        x = torch.max(x, dim=2)[0]
        x = self.fc(x)
        x = x.view(batch_size, num_seconds, -1)
       
        return x

def get_embedding_model(embedding_type, **kwargs):
    if embedding_type == 'mlp':
        return MLPEmbedding(**kwargs)
    elif embedding_type == 'pn':
        return PointNetEmbedding(**kwargs)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
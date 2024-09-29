import torch
import torch.nn as nn
import torch.nn.functional as F

# Adapter module to map facial landmarks and blendshapes to token embeddings
class MLPEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(MLPEmbedding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4096, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(4096, 3072, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(3072, embedding_dim, dtype=torch.bfloat16)
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        embedding = self.mlp(x)
        return embedding  # shape: (batch_size, embedding_dim)


# class CNNEmbedding(nn.Module):
#     def __init__(self, num_landmarks=478, seconds=30, embedding_dim):
#         super(FacialLandmarksToEmbedding, self).__init__()

#         # Convolution to process the 478 landmarks (3D coords) across frames
#         self.landmark_conv = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
#             nn.LeakyReLU(),
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
#             nn.LeakyReLU(),
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
#             nn.LeakyReLU(),
#             nn.AdaptiveAvgPool2d((num_landmarks, 1))  # Pool to reduce temporal dimension
#         )

#         # Convolution to process the blendshapes over frames
#         self.blendshape_conv = nn.Sequential(
#             nn.Conv1d(in_channels=num_blendshapes, out_channels=64, kernel_size=3, padding=1),
#             nn.LeakyReLU(),
#             nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.LeakyReLU(),
#             nn.AdaptiveAvgPool1d(1)  # Pool to reduce temporal dimension
#         )

#         # Linear projection for both landmarks and blendshapes
#         self.landmark_proj = nn.Linear(256 * num_landmarks, embedding_dim // 2)
#         self.blendshape_proj = nn.Linear(128, embedding_dim // 2)

#         # Final projection to token embedding space
#         self.final_proj = nn.Linear(embedding_dim, embedding_dim)

#     def forward(self, landmarks, blendshapes):
#         """
#         landmarks: Tensor of shape (batch_size, 478, 3, num_frames)
#         blendshapes: Tensor of shape (batch_size, 53, num_frames)
#         """
#         # Process the landmarks: (batch_size, 3, 478, num_frames) -> (batch_size, 256, 478, 1)
#         landmarks = landmarks.permute(0, 2, 1, 3)  # Permute to (batch_size, 3, 478, num_frames)
#         landmarks = self.landmark_conv(landmarks)
#         landmarks = landmarks.view(landmarks.size(0), -1)  # Flatten

#         # Process the blendshapes: (batch_size, 53, num_frames) -> (batch_size, 128, 1)
#         blendshapes = self.blendshape_conv(blendshapes)
#         blendshapes = blendshapes.view(blendshapes.size(0), -1)  # Flatten

#         # Project both to embedding space
#         landmarks_emb = self.landmark_proj(landmarks)
#         blendshapes_emb = self.blendshape_proj(blendshapes)

#         # Concatenate and project to final embedding space
#         combined = torch.cat((landmarks_emb, blendshapes_emb), dim=1)
#         output_embedding = self.final_proj(combined)

#         return output_embedding

def get_embedding_model(embedding_type, **kwargs):
    if embedding_type == 'mlp':
        return MLPEmbedding(**kwargs)
    # elif embedding_type == 'cnn':
    #     return CNNEmbedding(**kwargs)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from landmark_embeddings import get_embedding_model

# Custom llm with landmark prefix and emotion regression head
class CustomModelWithPrefix(nn.Module):
    def __init__(self, model_name, n_landmarks, embedding_type):
        super(CustomModelWithPrefix, self).__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # freeze llm
        for param in self.lm.parameters():
            param.requires_grad = False
        self.embedding_dim = self.lm.config.hidden_size

        self.landmark_embedding = get_embedding_model(embedding_type, input_dim=30*n_landmarks*3, embedding_dim=self.embedding_dim)

        # emotion regression head
        self.emotion_regressor = nn.Linear(self.embedding_dim, 6, dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask, landmarks):
        # landmarks_blendshapes shape: (batch_size, input_dim)
        landmark_embed = self.landmark_embedding(landmarks)  # shape: (batch_size, embedding_dim)
        inputs_embeds = self.lm.get_input_embeddings()(input_ids)  # shape: (batch_size, seq_len, embedding_dim)
        # Prepend landmark embedding
        landmark_embed = landmark_embed.unsqueeze(1)  # shape: (batch_size, 1, embedding_dim)
        inputs_embeds = torch.cat([landmark_embed, inputs_embeds], dim=1)  # shape: (batch_size, seq_len+1, embedding_dim)
        extended_attention_mask = torch.cat([torch.ones((attention_mask.size(0), 1), device=attention_mask.device), attention_mask], dim=1)
        # Pass through the language model
        outputs = self.lm(inputs_embeds=inputs_embeds, attention_mask=extended_attention_mask)
        # Extract the hidden state of the prepended token
        sequence_output = outputs.hidden_states[-1]  # shape: (batch_size, seq_len+1, embedding_dim)
        landmark_output = sequence_output[:, 0, :]  # shape: (batch_size, embedding_dim)
        # Emotion regression
        emotion_intensities = self.emotion_regressor(landmark_output)  # shape: (batch_size, 6)
        return emotion_intensities


        
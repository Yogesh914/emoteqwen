import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from landmark_embeddings import get_embedding_model

# Custom llm with landmark prefix and emotion regression head
class CustomModelWithPrefix(nn.Module):
    def __init__(self, model_name, n_landmarks, embedding_type):
        super(CustomModelWithPrefix, self).__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, output_hidden_states=True, return_dict_in_generate=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # freeze llm
        for param in self.lm.parameters():
            param.requires_grad = False
        self.embedding_dim = self.lm.config.hidden_size

        if embedding_type == 'mlp':
            self.landmark_embedding = get_embedding_model(embedding_type, input_dim=n_landmarks*3, embedding_dim=self.embedding_dim)
        elif embedding_type == 'pn':
            self.landmark_embedding = get_embedding_model(embedding_type, embedding_dim=self.embedding_dim)

        # emotion regression head
        self.emotion_regressor = nn.Sequential(
            nn.Linear(self.embedding_dim, 1024, dtype=torch.bfloat16),
            nn.GELU(), #gelu
            nn.Linear(1024, 512, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(512, 6, dtype=torch.bfloat16)
        )

    def forward(self, input_ids, attention_mask, landmarks):
        # landmarks_blendshapes shape: (batch_size, num_seconds, n_landmarks, 3)
        landmark_embed = self.landmark_embedding(landmarks)  # shape: (batch_size, num_seconds, embedding_dim)
        inputs_embeds = self.lm.get_input_embeddings()(input_ids)  # shape: (batch_size, seq_len, embedding_dim)

        # Prepend landmark embedding
        inputs_embeds = torch.cat([landmark_embed, inputs_embeds], dim=1)  # shape: (batch_size, num_seconds + seq_len, embedding_dim)
         
        
        landmarks_attention = torch.ones((attention_mask.size(0), landmark_embed.size(1)), device=attention_mask.device)
        extended_attention_mask = torch.cat([landmarks_attention, attention_mask], dim=1)


        # Pass through the language model
        outputs = self.lm(inputs_embeds=inputs_embeds, attention_mask=extended_attention_mask)
        #outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the hidden state of the prepended token
        sequence_output = outputs.hidden_states[-1]  # shape: (batch_size, seq_len+1, embedding_dim)
        landmark_output = sequence_output[:, -1, :]  # shape: (batch_size, embedding_dim)
        # landmark_output = torch.mean(sequence_output, dim=1)

        # Emotion regression
        emotion_intensities = self.emotion_regressor(landmark_output)  # shape: (batch_size, 6)
        return emotion_intensities


        
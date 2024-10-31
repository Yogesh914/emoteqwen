import torch
from torch.utils.data import Dataset, Subset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd

class EmotionDataset(Dataset):
    def __init__(self, transcriptions, landmarks, labels, tokenizer, mean, std, prompt):
        self.transcriptions = transcriptions
        self.landmarks = landmarks #(landmarks - mean) / std  # shape: (num_samples, 30, N_landmarks, 3)
        self.labels = labels  # shape: (num_samples, 6)
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.encodings = [self.tokenizer.encode_plus(
            transcription,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=True,
            return_tensors='pt',
        ) for transcription in self.transcriptions]

        assert len(self.landmarks) == len(self.labels) == len(self.transcriptions), "Mismatch in number of samples across files"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        landmark = self.landmarks[idx]
        label = self.labels[idx]
        encoding = self.encodings[idx]

        #full_text = f"Transcription: {transcription} \n {self.prompt}"

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'landmarks': torch.tensor(landmark, dtype=torch.bfloat16),
            'labels': torch.tensor(label, dtype=torch.bfloat16)
        }


def custom_collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    landmarks = torch.stack([item['landmarks'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'landmarks': landmarks,
        'labels': labels
    }

def load_emotion_dataset(transcriptions, landmarks, labels, tokenizer, mean=0, std=0, prompt=''):
    return EmotionDataset(transcriptions, landmarks, labels, tokenizer, mean, std, prompt)
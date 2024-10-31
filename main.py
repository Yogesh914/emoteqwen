from models import CustomModelWithPrefix
from dataset import load_emotion_dataset
from train import train_model
from evaluate import evaluate
from utils import set_seeds
from config import Config
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
import numpy as np
import os
import torch

def train_and_save_model(config):
    set_seeds(config.seed)
    model = CustomModelWithPrefix(config.model_name, config.n_landmarks, config.embedding_type)

    print(config.landmarks_file)
    transcriptions = np.load(config.transcriptions_file)
    landmarks = np.load(config.landmarks_file) 
    labels = np.load(config.labels_file)

    # full_dataset = load_emotion_dataset(
    #     landmarks=landmarks,
    #     transcriptions=transcriptions,
    #     labels=labels,
    #     tokenizer=model.tokenizer
    # )

    train_size = int((1 - (2 * config.eval_split)) * len(labels))
    val_size = int(config.eval_split * len(labels))
    eval_size = int(config.eval_split * len(labels))

    # train_dataset, val_dataset, _ = random_split(full_dataset, [0.8, 0.1, 0.1])

    # #norm_mean = np.mean(landmarks[:train_size], axis=0) 
    # #norm_std = np.std(landmarks[:train_size], axis=0) 

    train_dataset = load_emotion_dataset(
        landmarks=landmarks[:train_size],
        transcriptions=transcriptions[:train_size],
        labels=labels[:train_size],
        tokenizer=model.tokenizer
        #mean = norm_mean,
        #std = norm_std,
        #prompt = config.prompt
    )

    val_dataset = load_emotion_dataset(
        landmarks=landmarks[train_size:train_size + val_size],
        transcriptions=transcriptions[train_size:train_size + val_size],
        labels=labels[train_size:train_size + val_size],
        tokenizer=model.tokenizer
        #mean = norm_mean,
        #std = norm_std,
        #prompt = config.prompt
    )

    trained_model = train_model(model, train_dataset, val_dataset, config)

    save_path = os.path.join(config.model_save_dir, 'trained_model.pth')
    torch.save(trained_model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def load_and_evaluate_model(config):
    set_seeds(config.seed)
    model = CustomModelWithPrefix(config.model_name, config.n_landmarks, config.embedding_type)
    
    load_path = os.path.join(config.model_save_dir, 'best_model.pth')
    model.load_state_dict(torch.load(load_path))
    print(f"Model loaded from {load_path}")

    full_dataset = load_emotion_dataset(
        landmarks_file=config.landmarks_file,
        transcriptions_file=config.transcriptions_file,
        labels_file=config.labels_file,
        tokenizer=model.tokenizer
    )

    train_size = int((1 - config.eval_split) * len(full_dataset))
    eval_size = len(full_dataset) - train_size

    eval_dataset = Subset(full_dataset, range(train_size, train_size + eval_size))

    eval_loss, avg_errors = evaluate(model, eval_dataset, config)

    

if __name__ == "__main__":
    config = Config()
    train_and_save_model(config)
    #load_and_evaluate_model(config)
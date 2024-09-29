import torch
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os 
from datetime import datetime
from dataset import custom_collate_fn
from torch.utils.data import DataLoader

def mae_loss(predictions, targets):
    return torch.abs(predictions - targets).mean()

def train(model, data_loader, optimizer, device, epoch, writer):
    model = model.train()
    losses = []

    for batch_idx, d in enumerate(tqdm(data_loader)):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        landmarks = d["landmarks"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            landmarks=landmarks
        )

        loss = mae_loss(outputs, labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 10 == 0:
            writer.add_scalar('Training Loss', loss.item(), epoch * len(data_loader) + batch_idx)
        
        del input_ids, attention_mask, landmarks, labels, outputs, loss
        torch.cuda.empty_cache()

    return np.mean(losses)


def train_model(model, dataset, config):
    device = torch.device(config.device)
    model.to(device)

    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    
    optimizer = torch.optim.Adam([
        {'params': model.landmark_embedding.parameters()},
        {'params': model.emotion_regressor.parameters()}
    ], lr=config.learning_rate)

    log_dir = os.path.join(config.log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        train_loss = train(model, data_loader, optimizer, device, epoch, writer)
        print(f"Training Loss: {train_loss:.4f}")

        writer.add_scalar('Epoch Training Loss', train_loss, epoch)

    writer.close()
    print(f"Tensorboard logs saved to {log_dir}")
    return model


import torch
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os 
from datetime import datetime
from dataset import custom_collate_fn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

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

def validate(model, data_loader, device):
    model = model.eval()
    losses = []
    with torch.no_grad():
        for d in data_loader:
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
            
            del input_ids, attention_mask, landmarks, labels, outputs, loss
            torch.cuda.empty_cache()

    return np.mean(losses)


def train_model(model, train_dataset, val_dataset, config):
    device = torch.device(config.device)
    model.to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    
    optimizer = torch.optim.Adam([
        {'params': model.landmark_embedding.parameters()},
        {'params': model.emotion_regressor.parameters()}
    ], lr=config.learning_rate)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)


    log_dir = os.path.join(config.log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)

    best_val_loss = float('inf')
    # early_stopping_counter = 0
    # patience = 5

    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        train_loss = train(model, train_dataloader, optimizer, device, epoch, writer)
        val_loss = validate(model, val_dataloader, device)
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")



        writer.add_scalar('Epoch Training Loss', train_loss, epoch)
        writer.add_scalar('Epoch Validation Loss', val_loss, epoch)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.best_model)
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")

        # else:
        #     early_stopping_counter += 1
        #     if early_stopping_counter >= patience:
        #         print("Early stopping triggered!")
        #         break

        # if optimizer.param_groups[0]['lr'] < 1e-6:
        #     print("Learning rate too small. Stopping training.")
        #     break

    writer.close()
    print(f"Tensorboard logs saved to {log_dir}")
    return model

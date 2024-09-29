import torch
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np
import os 
from torch.utils.data import DataLoader
from dataset import custom_collate_fn

def mae_loss(predictions, targets):
    return torch.abs(predictions - targets).mean()

def evaluate_model(model, data_loader, device):
    model.eval()
    losses = []
    all_errors = []

    with torch.no_grad():
        for d in tqdm(data_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            landmarks = d["landmarks"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                landmarks=landmarks
            )

            loss = torch.nn.functional.mse_loss(outputs, labels)
            losses.append(loss.item)

            errors = torch.abs(outputs - labels)
            all_errors.append(errors)

    avg_loss = np.mean(losses)
    avg_errors = torch.cat(all_errors, dim=0).mean(dim=0).tolist()

    return avg_loss, avg_errors

def evaluate(model, eval_dataset, config):
    device = torch.device(config.device)
    model.to(device)

    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    eval_loss, avg_errors = evaluate_model(model, eval_loader, device)

    print(f"Evaluation Loss: {eval_loss:.4f}")
    print(f"Average Errors per Emotion: {avg_errors}")

    return eval_loss, avg_errors



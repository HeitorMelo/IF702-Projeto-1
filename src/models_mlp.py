import numpy as np
import torch
import torch.nn as nn
from src.utils import TimedModel

from tqdm import tqdm

class MLP(nn.Module):
    def __init__(self, input_size, num_classes, num_layers, neurons_per_layer, activation_function, dropout_rate):
        super(MLP, self).__init__()
        
        layers = []
        in_features = input_size
        
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, neurons_per_layer))
            layers.append(activation_function())
            if(dropout_rate > 0):
                layers.append(nn.Dropout(dropout_rate))
            in_features = neurons_per_layer
            
        layers.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def predict(self, x):
        with torch.no_grad():
            x = self.forward(x)
            return torch.argmax(x, dim=1)

def build_mlp(input_size, num_classes, num_layers, neurons_per_layer, activation_name,dropout_rate,):
    activations = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "Sigmoid": nn.Sigmoid}
    activation_function = activations[activation_name]
    
    return MLP(input_size, num_classes, num_layers, neurons_per_layer, activation_function,dropout_rate)

#### -------------------------

def train_epoch(model, dataloader, optimizer, criterion, criterion_name, epoch=0, total_epochs=10):
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)

    for images, labels in pbar:
        optimizer.zero_grad()
        outputs = model(images)
        
        if criterion_name == "MSELoss":
            labels_loss = torch.nn.functional.one_hot(labels, num_classes=10).float().to(images.device)
        else:
            labels_loss = labels
            
        loss = criterion(outputs, labels_loss)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / len(dataloader)
        
    return avg_loss

def validate_epoch(model, dataloader, criterion, criterion_name, epoch=0, total_epochs=10):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    timed_model = TimedModel(model)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Valid]", leave=False)
    
    with torch.no_grad():
        for images, labels in pbar:
            outputs = timed_model(images)
            
            if criterion_name == "MSELoss":
                labels_loss = torch.nn.functional.one_hot(labels, num_classes=10).float().to(images.device) 
            else:
                labels_loss = labels
                
            loss = criterion(outputs, labels_loss)
            running_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / len(dataloader)
    avg_time_per_batch = timed_model.total_time / len(dataloader)
            
    return avg_loss, avg_time_per_batch, all_labels, all_preds
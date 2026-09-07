import numpy as np
import torch
import torch.nn as nn
from src.utils import TimedModel
from tqdm import tqdm

class CNN(nn.Module):
    def __init__(self, 
                 num_classes=10, 
                 num_conv_layers=2, 
                 filters_base=32, 
                 kernel_size=3, 
                 stride=1, 
                 padding=1, 
                 pool_size=2, 
                 dropout_rate=0.3, 
                 activation_function=nn.ReLU):
        super(CNN, self).__init__()
        
        layers = []
        in_channels = 3  # Entrada RGB do CIFAR-10 (3x32x32)
        current_filters = filters_base
        current_spatial_size = 32  # Largura/Altura inicial
        
        # Bloco Convolucional Dinâmico
        for i in range(num_conv_layers):
            layers.append(nn.Conv2d(
                in_channels=in_channels, 
                out_channels=current_filters, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding
            ))
            layers.append(nn.BatchNorm2d(current_filters))
            layers.append(activation_function())
            
            # Recálculo do tamanho da imagem após Conv2d: floor((W - K + 2P)/S) + 1
            current_spatial_size = (current_spatial_size - kernel_size + 2 * padding) // stride + 1
            
            # Aplica MaxPool apenas se a imagem tiver tamanho suficiente
            if pool_size > 1 and current_spatial_size >= pool_size:
                layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_size))
                current_spatial_size = current_spatial_size // pool_size
            
            in_channels = current_filters
            current_filters *= 2  # Dobra o número de filtros a cada camada

        self.conv_block = nn.Sequential(*layers)
        
        # Garante que a dimensão não ficou <= 0 após as convoluções e poolings
        if current_spatial_size <= 0:
            raise ValueError(
                f"Combinação inválida de parâmetros (kernel_size={kernel_size}, stride={stride}, "
                f"padding={padding}, pool_size={pool_size}). A dimensão espacial reduziu para {current_spatial_size}."
            )

        last_num_filters = filters_base * (2 ** (num_conv_layers - 1))
        flatten_size = last_num_filters * current_spatial_size * current_spatial_size
        
        # Camadas Densas / Classificador Final
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 128),
            activation_function(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            x = self.forward(x)
            return torch.argmax(x, dim=1)

def build_cnn(num_classes=10, 
              num_conv_layers=2, 
              filters_base=32, 
              kernel_size=3, 
              stride=1, 
              padding=1, 
              pool_size=2, 
              dropout_rate=0.3, 
              activation_name="ReLU"):
    
    activations = {
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
        "GELU": nn.GELU,
        "Tanh": nn.Tanh
    }
    
    activation_function = activations.get(activation_name, nn.ReLU)
    
    return CNN(
        num_classes=num_classes,
        num_conv_layers=num_conv_layers,
        filters_base=filters_base,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        pool_size=pool_size,
        dropout_rate=dropout_rate,
        activation_function=activation_function
    )

#### -------------------------
# Funções de Treino e Validação

def train_epoch(model, dataloader, optimizer, criterion, criterion_name, epoch=0, total_epochs=10):
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train CNN]", leave=False)

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

    return running_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, criterion_name, epoch=0, total_epochs=10):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    timed_model = TimedModel(model)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Valid CNN]", leave=False)
    
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
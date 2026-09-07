import optuna
import wandb
import torch
import torch.nn as nn
import torch.optim as optim

from src.data_loader import get_dataloaders
from src.models_mlp import build_mlp, train_epoch, validate_epoch
from src.metrics import calculate_metrics

from src.utils import EarlyStopping



def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    num_conv_layers = trial.suggest_int("num_conv_layers", 1, 3)  # Tamanho da rede
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5])  # Janela de convolução
    stride = trial.suggest_categorical("stride", [1, 2])
    padding = trial.suggest_categorical("padding", [0, 1, 2])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    pool_size = trial.suggest_categorical("pool_size", [1, 2])  # 1 = sem pooling, 2 = pool 2x2
    
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    criterion_name = trial.suggest_categorical("criterion", ["CrossEntropyLoss", "MSELoss"])
    activation_name = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU", "GELU"])





    run_name = f"trial-{trial.number}_lr-{lr:.4f}_bs-{batch_size}"
    
    run = wandb.init(
        entity="Proj-IF702",
        project="miniprojeto1-cifar10",
        name=run_name,
        group="mlp_optimization",
        config={"lr": lr, "num_conv_layers": num_conv_layers,"kernel_size": kernel_size, "batch_size": batch_size, 
                "criterion": criterion_name, "activation": activation_name},
        reinit=True
    )
    
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size, is_mlp=False)

    try:
        model = build_cnn(
            num_classes=10,
            num_conv_layers=num_conv_layers,
            filters_base=32,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pool_size=pool_size,
            dropout_rate=dropout_rate,
            activation_name=activation_name
        )
    except ValueError:
        # Se a combinação de hiperparâmetros zerar/invalidar a imagem, o Optuna descarta a trial
        raise optuna.exceptions.TrialPruned()

    criterion = nn.MSELoss() if criterion_name == "MSELoss" else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ... [Restante do loop de épocas e WandB] ...
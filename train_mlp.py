import optuna
import wandb
import torch
import torch.nn as nn
import torch.optim as optim

from src.data_loader import get_dataloaders
from src.models_mlp import build_mlp, train_epoch, validate_epoch
from src.metrics import calculate_metrics

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    criterion_name = trial.suggest_categorical("criterion", ["CrossEntropyLoss", "MSELoss"])
    activation_name = trial.suggest_categorical("activation", ["ReLU", "Tanh"])

    run_name = f"trial-{trial.number}_lr-{lr:.4f}_bs-{batch_size}"

    run = wandb.init(
        project="miniprojeto1-cifar10",
        name=run_name,
        group="mlp_optimization",
        config={"lr": lr, "num_layers": num_layers, "batch_size": batch_size, 
                "criterion": criterion_name, "activation": activation_name},
        reinit=True
    )
    
    train_loader, val_loader = get_dataloaders(batch_size=batch_size, is_mlp=True)
    
    model = build_mlp(3072, 10, num_layers=num_layers, neurons_per_layer=64, activation_name=activation_name)
    
    criterion = nn.MSELoss() if criterion_name == "MSELoss" else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    epochs = 5
    inference_times = []
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, criterion_name, epoch + 1, epochs)
        
        val_loss, inference_time, labels, preds = validate_epoch(model, val_loader, criterion, criterion_name, epoch + 1, epochs)
        inference_times.append(inference_time)

        metrics = calculate_metrics(labels, preds)
        
        wandb.log({
            "epoch": epoch, 
            "train_loss": train_loss,
            "val_loss": val_loss, 
            "val_acc": metrics["acc_total"]
        })
    
            
    wandb.finish()
    
    return metrics["acc_total"], sum(inference_times) / len(inference_times)

if __name__ == "__main__":
    study = optuna.create_study(
        study_name="mlp-cifar10-multiobjective",
        storage="sqlite:///cifar10_optuna.db", 
        directions=["maximize", "minimize"],
        sampler=optuna.samplers.TPESampler(),
        load_if_exists=True
    )
    study.optimize(objective, n_trials=3)
    
    pareto_front_trials = study.best_trials

    print(f"Number of Pareto-optimal models found: {len(pareto_front_trials)}\n")

    for i, trial in enumerate(pareto_front_trials):
        print(f"--- Pareto Optimal Model {i+1} ---")
        
        accuracy = trial.values[0]
        inference_time = trial.values[1]
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Inference Time: {inference_time:.6f} sec/batch")
        print(f"Hyperparameters: {trial.params}\n")
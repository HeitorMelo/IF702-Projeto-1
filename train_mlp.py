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
    num_layers = trial.suggest_int("num_layers", 1, 3)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    criterion_name = trial.suggest_categorical("criterion", ["CrossEntropyLoss", "MSELoss"])
    activation_name = trial.suggest_categorical("activation", ["ReLU", "Tanh"])

    run_name = f"trial-{trial.number}_lr-{lr:.4f}_bs-{batch_size}"

    run = wandb.init(
        entity="Proj-IF702",
        project="miniprojeto1-cifar10",
        name=run_name,
        group="mlp_optimization",
        config={"lr": lr, "num_layers": num_layers, "batch_size": batch_size, 
                "criterion": criterion_name, "activation": activation_name},
        reinit=True
    )
    
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size, is_mlp=True)
    
    model = build_mlp(3072, 10, num_layers=num_layers, neurons_per_layer=64, activation_name=activation_name)
    
    criterion = nn.MSELoss() if criterion_name == "MSELoss" else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)



    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck'] 


    model_path = f"best_mlp_trial_{trial.number}.pth"
    early_stopping = EarlyStopping(patience=5, min_delta=1e-3,path=model_path)

    epochs = 15
    inference_times = []
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, criterion_name, epoch + 1, epochs)
        
        val_loss, inference_time, labels, preds = validate_epoch(model, val_loader, criterion, criterion_name, epoch + 1, epochs)
        inference_times.append(inference_time)

        metrics = calculate_metrics(labels, preds)
        
        log_data={
            "epoch": epoch, 
            "train_loss": train_loss,
            "val_loss": val_loss, 
            "val_acc": metrics["acc_total"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_inference_time_batch": inference_time

        }
        for class_idx, class_acc in metrics["acc_per_class"].items():
            class_label = class_names[class_idx]
            log_data[f"acc_class/{class_label}"] = class_acc


        wandb.log(log_data)

        early_stopping(val_loss,model)
        if early_stopping.early_stop:
            print(f"Early stopping ativado na época {epoch}")
            break

    # Regra importante: Carrega os pesos antes de ativar a ocntagem da paciencia
    model.load_state_dict(torch.load(model_path))

    # Plota a Matriz de Confusão Interativa do WandB ao final do Trial
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=labels,
            preds=preds,
            class_names=class_names
        )
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
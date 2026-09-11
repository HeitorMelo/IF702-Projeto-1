import os
import optuna
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.data_loader import get_dataloaders
from src.models_mlp import build_mlp, train_epoch, validate_epoch
from src.metrics import calculate_metrics, plot_confusion_matrix_figure

from src.utils import EarlyStopping


def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    criterion_name = trial.suggest_categorical("criterion", ["CrossEntropyLoss", "MSELoss"])
    activation_name = trial.suggest_categorical("activation", ["ReLU", "Tanh", "Sigmoid"])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.3)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    run_name = f"MLP_trial-{trial.number}_lr-{lr:.4f}_bs-{batch_size}"

    run = wandb.init(
        entity="Proj-IF702",
        project="miniprojeto1-cifar10",
        name=run_name,
        group="mlp_optimization",
        config={"lr": lr, "num_layers": num_layers, "batch_size": batch_size, 
                "criterion": criterion_name, "activation": activation_name, "dropout_rate":dropout_rate, "weight_decay":weight_decay},
        reinit=True
    )
    
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size, is_mlp=True)
    
    model = build_mlp(3072, 10, num_layers=num_layers, neurons_per_layer=64, activation_name=activation_name, dropout_rate=dropout_rate)
    
    criterion = nn.MSELoss() if criterion_name == "MSELoss" else nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)



    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck'] 


    model_path = os.path.join(wandb.run.dir, f"best_mlp_trial_{trial.number}.pth")
    early_stopping = EarlyStopping(patience=5, min_delta=1e-3,path=model_path)

    epochs = 15
    inference_times = []
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, criterion_name, epoch + 1, epochs)
        
        val_loss, inference_time, labels, preds = validate_epoch(model, val_loader, criterion, criterion_name, epoch + 1, epochs)
        inference_times.append(inference_time)

        metrics = calculate_metrics(labels, preds)
        
        log_data={
            "epoch": epoch + 1, 
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
            print(f"Early stopping ativado na época {epoch + 1}")
            break

    # Regra importante: Carrega os pesos antes de ativar a ocntagem da paciencia
    model.load_state_dict(torch.load(model_path))

    # 2. Recalcula a validação com os pesos do MELHOR modelo
    val_loss, inference_time, labels, preds = validate_epoch(
        model, val_loader, criterion, criterion_name, epoch + 1, epochs
    )
    best_metrics = calculate_metrics(labels, preds)

    # 3. Plota a figura estática em alta resolução para o WandB
    fig = plot_confusion_matrix_figure(best_metrics["confusion_matrix"], class_names)
    
    wandb.log({
        "confusion_matrix_img": wandb.Image(fig)
    })
    
    plt.close(fig)  # Libera a memória do Matplotlib

    wandb.save(model_path, base_path=run.dir) # save best model no wandb
    average_inference_time = sum(inference_times) / len(inference_times)
    trial.set_user_attr("inference_time", average_inference_time)
    wandb.finish()
    return best_metrics["acc_total"]

if __name__ == "__main__":
    study = optuna.create_study(
        study_name="mlp-cifar10-accuracy-v3",
        storage="sqlite:///cifar10_optuna.db", 
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        load_if_exists=True
    )
    study.optimize(objective, n_trials=20)
    
    pareto_front_trials = study.best_trials

    print(f"Number of Pareto-optimal models found: {len(pareto_front_trials)}\n")

    for i, trial in enumerate(pareto_front_trials):
        print(f"--- Pareto Optimal Model {i+1} ---")
        
        accuracy = trial.value
        inference_time = trial.user_attrs.get("inference_time")
        
        print(f"Accuracy: {accuracy:.4f}")
        if inference_time is not None:
            print(f"Inference Time: {inference_time:.6f} sec/batch")
        print(f"Hyperparameters: {trial.params}\n")
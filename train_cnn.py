import os
import optuna
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.data_loader import get_dataloaders
from src.models_cnn import build_cnn, train_epoch, validate_epoch
from src.metrics import calculate_metrics, plot_confusion_matrix_figure

from src.utils import EarlyStopping



def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    num_conv_layers = trial.suggest_categorical("num_conv_layers",[1, 3, 5, 7])  
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128,256])
    criterion_name = trial.suggest_categorical("criterion", ["CrossEntropyLoss", "MSELoss"])
    activation_name = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU", "GELU"]) # leaky aplica uma leve inclicnao pra valores negativos,gelu é otima no contexto imagem

    kernel_size = trial.suggest_categorical("kernel_size", [2,3,5])  # tamanho do filtro
    stride = trial.suggest_categorical("stride", [1, 2])
    padding = trial.suggest_categorical("padding", [0, 1, 2])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    pool_size = trial.suggest_categorical("pool_size", [1, 2])  # 1 = sem pooling, 2 = pool 2x2
    

    run_name = f"CNN_trial-{trial.number}_lr-{lr:.4f}_bs-{batch_size}"
    
    run = wandb.init(
        entity="Proj-IF702",
        project="miniprojeto1-cifar10",
        name=run_name,
        group="cnn_optimization",
        config={"lr": lr, "num_conv_layers": num_conv_layers, "kernel_size": kernel_size, "stride": stride, "padding": padding, "dropout_rate": dropout_rate,
            "pool_size": pool_size, "batch_size": batch_size,"criterion": criterion_name, "activation": activation_name},
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

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck'] 


    model_path = os.path.join(wandb.run.dir, f"best_cnn_trial_{trial.number}.pth")
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
            print(f"Early stopping ativado na época {epoch}")
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
    wandb.finish()
    return best_metrics["acc_total"], sum(inference_times) / len(inference_times)

if __name__ == "__main__":
    study = optuna.create_study(
        study_name="cnn-cifar10-multiobjective",
        storage="sqlite:///cifar10_optuna.db", 
        directions=["maximize", "minimize"],
        sampler=optuna.samplers.TPESampler(),
        load_if_exists=True
    )
    study.optimize(objective, n_trials=20)
    
    pareto_front_trials = study.best_trials

    print(f"Number of Pareto-optimal models found: {len(pareto_front_trials)}\n")

    for i, trial in enumerate(pareto_front_trials):
        print(f"--- Pareto Optimal Model {i+1} ---")
        
        accuracy = trial.values[0]
        inference_time = trial.values[1]
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Inference Time: {inference_time:.6f} sec/batch")
        print(f"Hyperparameters: {trial.params}\n")

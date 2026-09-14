import os
import optuna
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import get_dataloaders
from src.models_mlp import build_mlp
from src.models_cnn import build_cnn
from src.metrics import calculate_metrics, plot_confusion_matrix_figure
from src.utils import TimedModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_best_trial(study_name, db_path="sqlite:///cifar10_optuna.db"):
    """Carrega a melhor trial concluída de um estudo do Optuna."""
    study = optuna.load_study(study_name=study_name, storage=db_path)
    return study.best_trial

def find_checkpoint_path(trial_number, model_type):
    """Busca o arquivo .pth correspondente ao trial nos diretórios do WandB."""
    wandb_dir = "./wandb"
    filename_target = f"best_{model_type.lower()}_trial_{trial_number}.pth"
    
    for root, _, files in os.walk(wandb_dir):
        if filename_target in files:
            return os.path.join(root, filename_target)
    
    if os.path.exists(filename_target):
        return filename_target
        
    return None

def evaluate_on_test(model, test_loader, criterion_name="CrossEntropyLoss"):
    """Avalia o modelo no conjunto de teste."""
    model.eval()
    all_preds, all_labels = [], []
    timed_model = TimedModel(model)

    criterion = nn.MSELoss() if criterion_name == "MSELoss" else nn.CrossEntropyLoss()
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = timed_model(images)

            if criterion_name == "MSELoss":
                labels_loss = torch.nn.functional.one_hot(labels, num_classes=10).float().to(device)
                loss_inputs = torch.softmax(outputs, dim=1)
            else:
                labels_loss = labels
                loss_inputs = outputs

            loss = criterion(loss_inputs, labels_loss)
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(test_loader)
    avg_inference_time = timed_model.total_time / len(test_loader)
    metrics = calculate_metrics(all_labels, all_preds)
    
    return metrics, avg_loss, avg_inference_time

def evaluate_best_model(study_name, model_type):
    """Reconstrói e avalia o melhor modelo de um estudo (MLP ou CNN)."""
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

    print(f"\n================ AVALIANDO MELHOR MODELO - {model_type.upper()} ================")
    
    try:
        best_trial = load_best_trial(study_name)
    except Exception as e:
        print(f"Erro ao carregar o estudo '{study_name}': {e}")
        return None

    os.makedirs("test_results", exist_ok=True)
    params = best_trial.params
    checkpoint_path = find_checkpoint_path(best_trial.number, model_type)
    
    print(f"Melhor Trial: #{best_trial.number}")
    print(f"Val Accuracy (Optuna): {best_trial.value:.4f}")

    print(f"Chaves do Trial #{best_trial.number}:", params)

    
    # Garante is_mlp=True e o DataLoader adequado
    is_mlp = (model_type.upper() == "MLP")
    batch_size = params.get("batch_size", 128)
    _, _, test_loader = get_dataloaders(batch_size=batch_size, is_mlp=is_mlp)

    # 1. Reconstrói a arquitetura campeã usando os argumentos exatos do seu build_mlp
    if is_mlp:
        # Tenta pegar 'neurons_per_layer' do optuna, com fallback para 'hidden_dim' se usado previamente
        neurons = params.get("neurons_per_layer", params.get("hidden_dim", 128))
        
        model = build_mlp(
            input_size=3072,
            num_classes=10,
            num_layers=params.get("num_layers", 2),
            neurons_per_layer=neurons,
            activation_name=params.get("activation", "ReLU"),
            dropout_rate=params.get("dropout_rate", 0.2)
        ).to(device)
    else: # CNN
        # Captura os parâmetros exatos do Optuna com fallbacks para variações de nomes
        
       model = build_cnn(
            num_classes=10,
            num_conv_layers=params["num_conv_layers"],  # Acessa diretamente a chave 'num_conv_layers' (3)
            filters_base=32,
            kernel_size=params["kernel_size"],
            stride=params["stride"],
            padding=params["padding"],
            pool_size=params["pool_size"],
            dropout_rate=params["dropout_rate"],
            activation_name=params["activation"]
        ).to(device)


    # 2. Carrega os pesos salvos
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Carregando pesos de: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"⚠️ AVISO: Checkpoint não encontrado para Trial #{best_trial.number} ({checkpoint_path})!")


    # 3. Avalia no conjunto de Teste
    criterion_name = params.get("criterion", "CrossEntropyLoss")
    metrics, test_loss, inf_time = evaluate_on_test(model, test_loader, criterion_name)

    print(f"-> TEST Accuracy: {metrics['acc_total']:.4f}")
    print(f"-> TEST Loss: {test_loss:.4f}")
    print(f"-> Inference Time/Batch: {inf_time:.6f} s")

    # 4. Salva a Matriz de Confusão
    fig = plot_confusion_matrix_figure(metrics["confusion_matrix"], class_names)
    fig.suptitle(f"Melhor {model_type.upper()} (Trial #{best_trial.number}) - Test Acc: {metrics['acc_total']:.2%}")
    fig.savefig(f"test_results/cm_best_{model_type.lower()}_trial{best_trial.number}.png")
    plt.close(fig)

    return {
        "Model": model_type.upper(),
        "Trial": best_trial.number,
        "Val Acc (Optuna)": round(best_trial.value, 4),
        "Test Acc": round(metrics["acc_total"], 4),
        "Test Precision": round(metrics["precision"], 4),
        "Test Recall": round(metrics["recall"], 4),
        "Test Loss": round(test_loss, 4),
        "Inf Time (s/batch)": round(inf_time, 6),
        "Batch Size": params.get("batch_size"),
        "LR": params.get("lr"),
        "Activation": params.get("activation"),
        "Criterion": params.get("criterion")
    }

if __name__ == "__main__":
    STUDY_MLP = "mlp-cifar10-accuracy"
    STUDY_CNN = "cnn-cifar10-accuracy-"

    mlp_best = evaluate_best_model(STUDY_MLP, "MLP")
    cnn_best = evaluate_best_model(STUDY_CNN, "CNN")

    results = [res for res in [mlp_best, cnn_best] if res is not None]
    df = pd.DataFrame(results)

    print("\n================ COMPARAÇÃO DOS MELHORES MODELOS (TESTE) ================\n")
    print(df.to_string(index=False))

    df.to_csv("test_results/best_models_comparison.csv", index=False)
    print("\n✅ Resultados salvos em 'test_results/best_models_comparison.csv' e matrizes salvas em 'test_results/'.")
import os
import optuna
import wandb
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


ENTITY = "Proj-IF702"
PROJECT = "miniprojeto1-cifar10"

def fetch_best_run_from_wandb(group_name):
    """
    Busca todas as runs do grupo no WandB e retorna a com maior 'val_acc'.
    """
    api = wandb.Api()
    
    # Filtra apenas pelo grupo diretamente no servidor do WandB
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": group_name})
    
    best_run = None
    best_acc = -1.0
    
    for run in runs:
        val_acc = run.summary.get("val_acc", 0.0)
        if val_acc > best_acc:
            best_acc = val_acc
            best_run = run
            
    return best_run

def download_checkpoint_from_wandb(run, model_type):
    """
    Faz o download do arquivo .pth associado à run diretamente do servidor do WandB.
    """
    os.makedirs("./checkpoints", exist_ok=True)
    
    # Procura o arquivo .pth salvo nos arquivos da run
    target_file = None
    for file in run.files():
        if file.name.endswith(".pth") and model_type.lower() in file.name.lower():
            target_file = file
            break
            
    if target_file is None:
        # Fallback para qualquer arquivo .pth na run
        for file in run.files():
            if file.name.endswith(".pth"):
                target_file = file
                break

    if target_file is None:
        raise FileNotFoundError(f"Nenhum arquivo .pth encontrado na run {run.id} do WandB.")

    download_path = os.path.join("./checkpoints", os.path.basename(target_file.name))
    print(f"📥 Baixando '{target_file.name}' da run {run.id}...")
    target_file.download(root="./checkpoints", replace=True)
    print(f"✅ Checkpoint salvo localmente em: {download_path}")
    
    return download_path


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

def evaluate_best_model_wandb(group_name, model_type):
    """Obtém a melhor run do WandB Online, baixa o checkpoint e avalia no Teste."""
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

    print(f"\n================ AVALIANDO MELHOR MODELO ONLINE ({model_type.upper()}) ================")
    
    try:
        run = fetch_best_run_from_wandb(group_name)
        if not run:
            print(f"❌ Nenhuma run encontrada para o grupo '{group_name}'.")
            return None
    except Exception as e:
        print(f"❌ Erro ao conectar com a API do WandB: {e}")
        return None

    os.makedirs("test_results", exist_ok=True)
    config = run.config
    val_acc = run.summary.get("val_acc", 0.0)
    
    print(f"Run Campeã WandB: {run.name} (ID: {run.id}) | Criador: {run.user.username if hasattr(run, 'user') else user_login}")
    print(f"Val Accuracy (WandB Summary): {val_acc:.4f}")
    print("Config do WandB Online:", config)

    # 1. Baixa o checkpoint diretamente do WandB
    try:
        checkpoint_path = download_checkpoint_from_wandb(run, model_type)
    except Exception as e:
        print(f"❌ Erro no download do modelo: {e}")
        return None

    is_mlp = (model_type.upper() == "MLP")
    batch_size = config.get("batch_size", 64)
    _, _, test_loader = get_dataloaders(batch_size=batch_size, is_mlp=is_mlp)

    # 2. Instancia o modelo com a config gravada na nuvem
    if is_mlp:
        neurons = config.get("neurons_per_layer", config.get("hidden_dim", 256))
        model = build_mlp(
            input_size=3072,
            num_classes=10,
            num_layers=config.get("num_layers", 1),
            neurons_per_layer=neurons,
            activation_name=config.get("activation", "ReLU"),
            dropout_rate=config.get("dropout_rate", 0.05)
        ).to(device)
    else: # CNN
        model = build_cnn(
            num_classes=10,
            num_conv_layers=config.get("num_conv_layers", 3),
            filters_base=config.get("filters_base", 32),
            kernel_size=config.get("kernel_size", 3),
            stride=config.get("stride", 1),
            padding=config.get("padding", 1),
            pool_size=config.get("pool_size", 2),
            dropout_rate=config.get("dropout_rate", 0.2),
            activation_name=config.get("activation", "GELU")
        ).to(device)

    # 3. Carrega os pesos no modelo instanciado
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 4. Avalia no conjunto de Teste
    criterion_name = config.get("criterion", "CrossEntropyLoss")
    metrics, test_loss, inf_time = evaluate_on_test(model, test_loader, criterion_name)

    print(f"-> TEST Accuracy: {metrics['acc_total']:.4f}")
    print(f"-> TEST Loss: {test_loss:.4f}")
    print(f"-> Inference Time/Batch: {inf_time:.6f} s")

    # 5. Salva a Matriz de Confusão
    fig = plot_confusion_matrix_figure(metrics["confusion_matrix"], class_names)
    # Limpa títulos anteriores da figura para não sobrepor
    for ax in fig.get_axes():
        ax.set_title("") 
    
    # Define um título único, limpo e com espaço adequado (y=1.02)
    fig.suptitle(
        f"Melhor {model_type.upper()} ({run.name}) - Test Acc: {metrics['acc_total']:.2%}",
        fontsize=12,
        y=0.98
    )
    
    # Ajusta o layout para garantir que nada fique cortado ou sobreposto
    fig.tight_layout()
    fig.savefig(f"test_results/cm_best_{model_type.lower()}_{run.id}.png", bbox_inches='tight', dpi=300)
    plt.close(fig)
    return {
        "Model": model_type.upper(),
        "WandB Run ID": run.id,
        "Val Acc (WandB)": round(val_acc, 4),
        "Test Acc": round(metrics["acc_total"], 4),
        "Test Precision": round(metrics["precision"], 4),
        "Test Recall": round(metrics["recall"], 4),
        "Test Loss": round(test_loss, 4),
        "Inf Time (s/batch)": round(inf_time, 6),
        "Batch Size": config.get("batch_size"),
        "LR": config.get("lr"),
        "Activation": config.get("activation"),
        "Criterion": config.get("criterion")
    }


if __name__ == "__main__":
    GROUP_MLP = "mlp_optimization" 
    GROUP_CNN = "cnn_optimization"

    mlp_best = evaluate_best_model_wandb(GROUP_MLP, "MLP")
    cnn_best = evaluate_best_model_wandb(GROUP_CNN, "CNN")

    results = [res for res in [mlp_best, cnn_best] if res is not None]
    
    if results:
        df = pd.DataFrame(results)
        print("\n================ COMPARAÇÃO DOS MELHORES MODELOS (TESTE) ================\n")
        print(df.to_string(index=False))

        df.to_csv("test_results/best_models_comparison.csv", index=False)
        print("\n✅ Resultados salvos em 'test_results/best_models_comparison.csv' e matrizes salvas em 'test_results/'.")
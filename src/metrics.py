import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc_total = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    # Acurácia por Classe
    cm = confusion_matrix(y_true, y_pred)
    
    acertos_por_classe = cm.diagonal()
    total_por_classe = cm.sum(axis=1)
    
    acc_array = np.divide(acertos_por_classe, total_por_classe, 
                          out=np.zeros_like(acertos_por_classe, dtype=float), 
                          where=total_por_classe != 0)
    
    acc_per_class = {i: float(acc_array[i]) for i in range(len(acc_array))}

    return {
        "acc_total": float(acc_total),
        "precision": float(precision),
        "recall": float(recall),
        "acc_per_class": acc_per_class
    }
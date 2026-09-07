import time
import torch

class TimedModel:
    def __init__(self, model):
        self.model = model
        self.total_time = 0.0

    def __call__(self, *args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        outputs = self.model(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.total_time += (time.perf_counter() - start)
        
        return outputs
        
    def reset(self):
        self.total_time = 0.0


class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4,path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss,model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Salva o modelo sempre que a validação melhorar
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
    def save_checkpoint(self, model):
        """Salva os pesos do modelo no caminho especificado."""
        torch.save(model.state_dict(), self.path)
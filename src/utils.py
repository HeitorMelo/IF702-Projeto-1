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
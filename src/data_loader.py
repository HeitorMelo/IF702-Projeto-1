import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(batch_size=32, is_mlp=False):
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    
    if is_mlp:
        transform_list.append(transforms.Lambda(lambda x: torch.flatten(x)))
        
    transform = transforms.Compose(transform_list)

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)



    # Divide as 50.000 imagens em Treino (40.000) e Validação (10.000)
    generator = torch.Generator().manual_seed(42) # Garante que a divisão seja idêntica sempre
    train_subset, val_subset = random_split(
        train_dataset,
        [40000, 10000],
        generator=generator
    )


    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
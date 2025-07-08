from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms    
import random
import numpy as np

def load_cifar100_cl_tasks(num_tasks=5, batch_size=64, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])

    train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    all_classes = list(range(100))
    random.shuffle(all_classes)

    # Split into equal class groups
    class_splits = [all_classes[i * 20:(i + 1) * 20] for i in range(num_tasks)]

    task_dataloaders = []

    for task_classes in class_splits:
        train_indices = [i for i, (_, label) in enumerate(train_data) if label in task_classes]
        test_indices  = [i for i, (_, label) in enumerate(test_data) if label in task_classes]

        train_loader = DataLoader(Subset(train_data, train_indices), batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(Subset(test_data, test_indices), batch_size=batch_size, shuffle=False)

        task_dataloaders.append((train_loader, test_loader))

    return task_dataloaders

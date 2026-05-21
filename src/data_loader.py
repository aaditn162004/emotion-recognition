# src/data_loader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size):
    """
    Creates PyTorch DataLoaders from image folders.

    Args:
        data_dir (string): Path to the main data directory (containing 'train' and 'test' folders).
        batch_size (int): The number of samples per batch.

    Returns:
        A tuple containing the training DataLoader, testing DataLoader, and class names.
    """
    
    # Define image transformations. This includes converting to grayscale,
    # resizing, converting to a tensor, and normalizing.
    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # Normalizing for a single channel
    ])

    # Create ImageFolder datasets
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=data_transforms)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=data_transforms)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Get class names for reference
    class_names = train_dataset.classes
    print(f"Detected classes: {class_names}")

    return train_loader, test_loader, class_names
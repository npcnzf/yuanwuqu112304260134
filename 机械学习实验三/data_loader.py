import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class MNISTDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28).astype(np.float32) / 255.0
        image = image[np.newaxis, :, :]

        if self.labels is not None:
            label = self.labels[idx]
            return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        return torch.tensor(image, dtype=torch.float32)


def load_data(train_path, test_path, val_size=0.1, random_state=42):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X = train_df.iloc[:, 1:].values
    y = train_df.iloc[:, 0].values
    X_test = test_df.values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state, stratify=y
    )

    return X_train, X_val, y_train, y_val, X_test


def get_data_loaders(X_train, X_val, y_train, y_val, X_test,
                     batch_size=64, use_augmentation=False, num_workers=0):
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ToTensor(),
        ])
    else:
        train_transform = None

    train_dataset = MNISTDataset(X_train, y_train, transform=train_transform)
    val_dataset = MNISTDataset(X_val, y_val)
    test_dataset = MNISTDataset(X_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
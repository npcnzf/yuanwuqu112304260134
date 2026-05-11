import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from data_loader import load_data, get_data_loaders
from model import CNNModel, save_model, load_model


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='model_best.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        save_model(model, self.path)
        self.val_loss_min = val_loss


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    return train_loss, train_acc


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    return val_loss, val_acc


def predict_test(model, X_test, device, batch_size=64):
    model.eval()
    _, _, test_loader = get_data_loaders(None, None, None, None, X_test, batch_size=batch_size)
    
    predictions = []
    with torch.no_grad():
        for images in tqdm(test_loader, desc='Predicting'):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
    
    return predictions


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\nLoading data...")
    X_train, X_val, y_train, y_val, X_test = load_data(
        'train.csv', 'test.csv', val_size=0.1
    )
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    
    model = CNNModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()
    
    train_loader, val_loader, _ = get_data_loaders(
        X_train, X_val, y_train, y_val, None,
        batch_size=64,
        use_augmentation=True
    )
    
    early_stopping = EarlyStopping(patience=10, verbose=True, path='model_best.pth')
    
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    
    print(f"\n{'='*60}")
    print("TRAINING HIGH-PERFORMANCE CNN MODEL")
    print(f"{'='*60}")
    
    epochs = 100
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f'New best validation accuracy: {best_val_acc:.2f}%')
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
    
    model = load_model(model, 'model_best.pth', device)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    print("\nGenerating test predictions...")
    test_predictions = predict_test(model, X_test, device)
    
    submission_df = pd.DataFrame({
        'ImageId': range(1, len(test_predictions) + 1),
        'Label': test_predictions
    })
    submission_df.to_csv('submission.csv', index=False)
    print("Submission file saved to submission.csv")
    
    print("\nModel training complete! You can now run 'python app.py' to start the web application.")


if __name__ == "__main__":
    main()
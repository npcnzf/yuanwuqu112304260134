import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm

from data_loader import load_data, get_data_loaders
from model import CNNModel, save_model, load_model


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='best_model.pth'):
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


def train_model(config, X_train, X_val, y_train, y_val, device, save_best=False):
    model = CNNModel().to(device)
    
    if config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    elif config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    criterion = nn.NLLLoss()
    
    train_loader, val_loader, _ = get_data_loaders(
        X_train, X_val, y_train, y_val, None,
        batch_size=config['batch_size'],
        use_augmentation=config['use_augmentation']
    )
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    early_stopping = None
    if config['use_early_stopping']:
        early_stopping = EarlyStopping(
            patience=10, verbose=True,
            path=config['model_path'] if save_best else 'temp_model.pth'
        )
    
    print(f"\n{'='*50}")
    print(f"Training with config: {config['name']}")
    print(f"{'='*50}")
    
    for epoch in range(config['epochs']):
        print(f'\nEpoch {epoch+1}/{config["epochs"]}')
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')
        
        if early_stopping:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break
    
    if early_stopping:
        model = load_model(model, early_stopping.path, device)
        if not save_best:
            os.remove(early_stopping.path)
    
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'final_train_acc': train_accs[-1],
        'final_val_acc': val_accs[-1],
        'min_train_loss': min(train_losses),
        'min_val_loss': min(val_losses),
        'converged_epoch': len(train_losses)
    }
    
    return model, results


def run_comparison_experiments(X_train, X_val, y_train, y_val, device):
    experiments = [
        {
            'name': 'Exp1_SGD_0.01_64_noaug_noes',
            'optimizer': 'SGD',
            'lr': 0.01,
            'batch_size': 64,
            'epochs': 50,
            'use_augmentation': False,
            'use_early_stopping': False,
            'model_path': 'model_exp1.pth'
        },
        {
            'name': 'Exp2_Adam_0.001_64_noaug_noes',
            'optimizer': 'Adam',
            'lr': 0.001,
            'batch_size': 64,
            'epochs': 50,
            'use_augmentation': False,
            'use_early_stopping': False,
            'model_path': 'model_exp2.pth'
        },
        {
            'name': 'Exp3_Adam_0.001_128_noaug_yeses',
            'optimizer': 'Adam',
            'lr': 0.001,
            'batch_size': 128,
            'epochs': 50,
            'use_augmentation': False,
            'use_early_stopping': True,
            'model_path': 'model_exp3.pth'
        },
        {
            'name': 'Exp4_Adam_0.001_64_yesaug_yeses',
            'optimizer': 'Adam',
            'lr': 0.001,
            'batch_size': 64,
            'epochs': 50,
            'use_augmentation': True,
            'use_early_stopping': True,
            'model_path': 'model_exp4.pth'
        }
    ]
    
    all_results = {}
    all_models = {}
    
    for config in experiments:
        model, results = train_model(config, X_train, X_val, y_train, y_val, device, save_best=True)
        all_results[config['name']] = results
        all_models[config['name']] = model
    
    return all_results, all_models


def plot_loss_comparison(all_results, save_path='loss_comparison.png'):
    plt.figure(figsize=(12, 6))
    
    for name, results in all_results.items():
        plt.plot(results['train_losses'], label=f'{name} (Train)', linewidth=2)
        plt.plot(results['val_losses'], label=f'{name} (Val)', linestyle='--', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss comparison plot saved to {save_path}")
    plt.show()


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
    
    print("\nRunning comparison experiments...")
    all_results, all_models = run_comparison_experiments(X_train, X_val, y_train, y_val, device)
    
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*60)
    for name, results in all_results.items():
        print(f"\n{name}:")
        print(f"  Final Train Acc: {results['final_train_acc']:.2f}%")
        print(f"  Final Val Acc: {results['final_val_acc']:.2f}%")
        print(f"  Min Train Loss: {results['min_train_loss']:.4f}")
        print(f"  Min Val Loss: {results['min_val_loss']:.4f}")
        print(f"  Converged at Epoch: {results['converged_epoch']}")
    
    plot_loss_comparison(all_results)
    
    print("\n" + "="*60)
    print("TRAINING FINAL HIGH-PERFORMANCE MODEL")
    print("="*60)
    
    final_config = {
        'name': 'Final_Model',
        'optimizer': 'Adam',
        'lr': 0.001,
        'batch_size': 64,
        'epochs': 100,
        'use_augmentation': True,
        'use_early_stopping': True,
        'model_path': 'model_best.pth'
    }
    
    final_model, final_results = train_model(
        final_config, X_train, X_val, y_train, y_val, device, save_best=True
    )
    
    print(f"\nFinal Model Results:")
    print(f"  Best Val Acc: {max(final_results['val_accs']):.2f}%")
    
    print("\nGenerating test predictions...")
    test_predictions = predict_test(final_model, X_test, device)
    
    submission_df = pd.DataFrame({
        'ImageId': range(1, len(test_predictions) + 1),
        'Label': test_predictions
    })
    submission_df.to_csv('submission.csv', index=False)
    print("Submission file saved to submission.csv")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    import pandas as pd
    main()
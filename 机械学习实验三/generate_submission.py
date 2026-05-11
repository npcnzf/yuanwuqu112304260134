import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

from data_loader import load_data, get_data_loaders
from model import CNNModel, load_model


def predict_test(model, X_test, device, batch_size=64):
    print(f"Creating test dataloader with {len(X_test)} samples...")
    model.eval()

    test_dataset = type('TestDataset', (), {
        '__len__': lambda self: len(X_test),
        '__getitem__': lambda self, idx: (
            torch.tensor(X_test[idx].reshape(28, 28).astype(np.float32) / 255.0)[np.newaxis, :, :],
        )
    })()

    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    predictions = []
    print("Starting prediction...")
    with torch.no_grad():
        for i, images in enumerate(tqdm(test_loader, desc='Predicting')):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            if i == 0:
                print(f"First batch predicted: {predicted[:5].tolist()}")

    return predictions


def main():
    print("="*60)
    print("GENERATING SUBMISSION FILE")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n[1/4] Loading data...")
    try:
        X_train, X_val, y_train, y_val, X_test = load_data(
            'train.csv', 'test.csv', val_size=0.1
        )
        print(f"   Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    except Exception as e:
        print(f"   ERROR loading data: {e}")
        sys.exit(1)

    print("\n[2/4] Loading model...")
    try:
        model = CNNModel().to(device)
        model = load_model(model, 'model_best.pth', device)
        print("   Model loaded successfully!")
    except Exception as e:
        print(f"   ERROR loading model: {e}")
        print("   Make sure model_best.pth exists (run train_fast.py first)")
        sys.exit(1)

    print("\n[3/4] Generating test predictions...")
    try:
        test_predictions = predict_test(model, X_test, device)
        print(f"   Generated {len(test_predictions)} predictions")
    except Exception as e:
        print(f"   ERROR during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n[4/4] Saving submission.csv...")
    try:
        submission_df = pd.DataFrame({
            'ImageId': range(1, len(test_predictions) + 1),
            'Label': test_predictions
        })
        submission_df.to_csv('submission.csv', index=False)
        print(f"   Saved to submission.csv!")
    except Exception as e:
        print(f"   ERROR saving file: {e}")
        sys.exit(1)

    print("\n" + "="*60)
    print("SUBMISSION FILE GENERATED SUCCESSFULLY!")
    print("="*60)
    print(f"\nFirst 10 predictions:")
    print(submission_df.head(10))
    print(f"\nLabel distribution:")
    print(submission_df['Label'].value_counts().sort_index())


if __name__ == "__main__":
    main()
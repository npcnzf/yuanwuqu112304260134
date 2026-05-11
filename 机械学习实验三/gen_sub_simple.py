import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

print("Loading data...")
test_df = pd.read_csv('test.csv')
X_test = test_df.values
print(f"Test samples: {len(X_test)}")

print("Loading model...")
from model import CNNModel, load_model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel().to(device)
model = load_model(model, 'model_best.pth', device)
print("Model loaded!")

print("Predicting...")
predictions = []
model.eval()
with torch.no_grad():
    for i in tqdm(range(len(X_test)), desc="Predicting"):
        img = X_test[i]
        img_tensor = torch.tensor(img.reshape(1, 1, 28, 28).astype(np.float32) / 255.0).to(device)
        output = model(img_tensor)
        _, predicted = torch.max(output.data, 1)
        predictions.append(predicted.item())

print(f"Total predictions: {len(predictions)}")

print("Saving submission.csv...")
submission_df = pd.DataFrame({
    'ImageId': range(1, len(predictions) + 1),
    'Label': predictions
})
submission_df.to_csv('submission.csv', index=False)
print("Done!")
print(f"\nFirst 10 predictions:")
print(submission_df.head(10))
print(f"\nLast 10 predictions:")
print(submission_df.tail(10))
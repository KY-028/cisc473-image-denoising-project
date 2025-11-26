"""
NAFNet-small Training Script
Trains a lightweight NAFNet model on the BSD68 and BSDS300 datasets for image denoising.
Saves the best model based on validation loss and plots training curves.

Usage: python -m src.train.train_nafnet
"""

import os
import time
import torch
import pickle
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import NAFNet-small model
from ..models.nafnet import NAFNet
# Import dataset loader
from ..data.denoise_dataset import DenoiseDataset

# Prepare directories
current_dir = os.path.dirname(__file__)
save_dir = os.path.join(current_dir, '..', 'checkpoints')
os.makedirs(save_dir, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load dataset
dataset = DenoiseDataset(dirs=["BSD68", "BSDS300"], size=128, sigma=25/255.0)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

# DataLoaders
trainloader = DataLoader(train_set, batch_size=16, shuffle=True)
valloader = DataLoader(val_set, batch_size=16, shuffle=False)

# Build lightweight NAFNet-small
model = NAFNet().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Checkpoint paths
best_model_path = os.path.join(save_dir, "nafnet_small_best.pth")
history_path = os.path.join(save_dir, "nafnet_small_history.pkl")

# Load previous checkpoint if exists
best_val_loss = float('inf')
train_loss_list, val_loss_list = [], []

if os.path.exists(best_model_path):
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    try:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    except:
        print("Scheduler incompatible, resetting scheduler.")
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    print("Loaded existing model from previous run.")

# Load history if exists
if os.path.exists(history_path):
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
        train_loss_list = history.get('train_loss', [])
        val_loss_list = history.get('val_loss', [])
    print(f"Loaded previous training history ({len(train_loss_list)} epochs).")

# Training setup
epochs = 100
patience = 10
no_improve = 0
start_epoch = len(train_loss_list)

print(f"Starting from epoch {start_epoch} with best_val_loss={best_val_loss:.6f}")
start_time = time.time()

# ============================
#        Training Loop
# ============================
for epoch in range(start_epoch, start_epoch + epochs):

    # ---- Train ----
    model.train()
    train_loss = 0.0
    for noisy, clean in trainloader:
        noisy, clean = noisy.to(device), clean.to(device)
        
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(trainloader)
    train_loss_list.append(avg_train_loss)

    # ---- Validate ----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for noisy, clean in valloader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            val_loss += criterion(output, clean).item()

    avg_val_loss = val_loss / len(valloader)
    val_loss_list.append(avg_val_loss)
    scheduler.step(avg_val_loss)

    print(f"Epoch [{epoch+1}/{start_epoch + epochs}]  "
          f"Train Loss: {avg_train_loss:.6f}  Val Loss: {avg_val_loss:.6f}")

    # ---- Save best model ----
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss
        }, best_model_path)
        print(" New best NAFNet model saved.")
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping triggered.")
            break

# ---- Save history ----
history = {'train_loss': train_loss_list, 'val_loss': val_loss_list}
with open(history_path, 'wb') as f:
    pickle.dump(history, f)

# ---- Plot training curve ----
plt.figure()
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('NAFNet-small Training Curve')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "nafnet_small_training_curve.png"))

# ---- Total time ----
elapsed = (time.time() - start_time) / 60
print(f"Training complete. Best Val Loss: {best_val_loss:.6f}. Time: {elapsed:.2f} minutes.")

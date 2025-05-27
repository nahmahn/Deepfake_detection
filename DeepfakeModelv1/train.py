import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import DeepfakeDataset
from model import SVFTSwinFusion
from torch import nn, optim
from tqdm import tqdm
import os
from torch.utils.data import random_split
import matplotlib.pyplot as plt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_set = DeepfakeDataset("frames/train", transform)
    val_ratio = 0.2
    val_size = int(len(train_set) * val_ratio)
    train_size = len(train_set) - val_size
    train_subset, val_subset = random_split(train_set, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model = SVFTSwinFusion().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    checkpoint_path = "svft_swin_v1.pth"
    start_epoch = 1

    best_val_loss = float('inf')
    patience = 3
    counter = 0

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(start_epoch, start_epoch + 10):
        model.train()
        total_loss, correct = 0, 0

        for i, (images, spectral, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            print("images shape:", images.shape)
            print("spectral shape:", spectral.shape)
            print("labels shape:", labels.shape)
            images = images.to(device)
            spectral = spectral.to(device)
            labels = labels.to(device)
            logits = model(images, spectral)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()

            if i % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        acc = correct / len(train_subset)
        print(f"\nEpoch {epoch+1} Done | Loss: {total_loss:.4f} | Accuracy: {acc:.4f}\n")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, spectral, labels in val_loader:
                images = images.to(device)
                spectral = spectral.to(device)
                labels = labels.to(device)
                logits = model(images, spectral)
                loss = criterion(logits, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), checkpoint_path)  
            print(f" Model weights saved to {checkpoint_path}")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    torch.save(model.state_dict(), checkpoint_path)
    print(f" Model weights saved to {checkpoint_path}")

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.show()

    plt.figure()
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    plt.show()

if __name__ == "__main__":
    main()

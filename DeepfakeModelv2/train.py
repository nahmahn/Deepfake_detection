import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DeepfakeDataset
from model import DeepfakeBaseline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def create_labels_dict(root_dir):
    labels_dict = {}
    for label_name in ['real', 'fake']:
        class_dir = os.path.join(root_dir, label_name)
        if not os.path.exists(class_dir):
            continue
        for video_name in os.listdir(class_dir):
            video_path = os.path.join(label_name, video_name)  
            full_video_path = os.path.join(root_dir, video_path)
            if os.path.isdir(full_video_path):
                labels_dict[video_path] = 0 if label_name == 'real' else 1
    return labels_dict


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    loop = tqdm(dataloader, desc='Training', leave=False)
    for frames, labels in loop:
        frames, labels = frames.to(device), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(frames).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * frames.size(0)

        loop.set_postfix(loss=loss.item())
    return running_loss / len(dataloader.dataset)


def validate(model, dataloader, device):
    model.eval()
    preds, targets = [], []
    loop = tqdm(dataloader, desc='Validation', leave=False)
    with torch.no_grad():
        for frames, labels in loop:
            frames, labels = frames.to(device), labels.to(device).float()
            outputs = model(frames).squeeze()
            preds.append(torch.sigmoid(outputs).cpu())
            targets.append(labels.cpu())
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    auc = roc_auc_score(targets, preds)
    return auc


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(torch.cuda.is_available())

    root_dir = 'data/faces/train' 
    labels_dict = create_labels_dict(root_dir)

    video_names = list(labels_dict.keys())
    train_videos, val_videos = train_test_split(
        video_names,
        test_size=0.2,
        random_state=42,
        stratify=[labels_dict[v] for v in video_names]
    )

    train_labels = {v: labels_dict[v] for v in train_videos}
    val_labels = {v: labels_dict[v] for v in val_videos}

    train_dataset = DeepfakeDataset(root_dir, train_labels)
    val_dataset = DeepfakeDataset(root_dir, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    model = DeepfakeBaseline().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_auc = 0
    patience = 5
    no_improvement_count = 0
    epochs = 20

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_auc = validate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f} - Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'best_model.pth')
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print("Early stopping triggered.")
            break


if __name__ == '__main__':
    main()

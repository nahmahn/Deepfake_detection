import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DeepfakeDataset
from model import DeepfakeModel
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
        frames = frames.to(device)
        labels = labels.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(frames)
        if outputs.dim() > 1:
            outputs = outputs.squeeze(1)
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
            frames = frames.to(device)
            labels = labels.to(device).float()
            outputs = model(frames)
            if outputs.dim() > 1:
                outputs = outputs.squeeze(1)
            preds.append(torch.sigmoid(outputs).cpu())
            targets.append(labels.cpu())
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    auc = roc_auc_score(targets, preds)
    return auc


def set_backbone_requires_grad(model, requires_grad):
    for param in model.backbone.parameters():
        param.requires_grad = requires_grad

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

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

    train_dataset = DeepfakeDataset(root_dir, train_labels, num_frames=20, train=True)
    val_dataset = DeepfakeDataset(root_dir, val_labels, num_frames=20, train= False)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    model = DeepfakeModel().to(device)
    
    train_iter = iter(train_loader)
    frames, labels = next(train_iter)
    print(f"Frames shape: {frames.shape}")
    print(f"Frames device: {frames.device}")
    print(f"Model device: {next(model.parameters()).device}")

    set_backbone_requires_grad(model, False)

    lstm_params = list(model.lstm.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.AdamW(lstm_params, lr=1e-3, weight_decay=1e-4)  

    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0
    patience = 5
    no_improvement_count = 0
    epochs = 20

    unfreeze_epoch = 5

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        if epoch == unfreeze_epoch:
            print("Unfreezing backbone and lowering learning rate.")
            set_backbone_requires_grad(model, True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

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

import os
import torch
from torch.utils.data import DataLoader
from dataset import DeepfakeDataset
from model import DeepfakeModel
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from tqdm import tqdm
import numpy as np

def find_best_threshold(targets, preds):
    best_thresh = 0.5
    best_f1 = 0

    for t in np.linspace(0.1, 0.9, 81):  
        pred_labels = (preds >= t).astype(int)
        f1 = f1_score(targets, pred_labels)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1

def evaluate(model, dataloader, device):
    model.eval()
    preds, targets = [], []
    loop = tqdm(dataloader, desc='Evaluating', leave=False)
    with torch.no_grad():
        for frames, labels in loop:
            frames = frames.to(device)
            labels = labels.to(device).float()
            outputs = model(frames)
            if outputs.dim() > 1:
                outputs = outputs.squeeze(1)
            probs = torch.sigmoid(outputs).cpu()
            preds.append(probs)
            targets.append(labels.cpu())

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    auc = roc_auc_score(targets, preds)
    best_thresh, best_f1 = find_best_threshold(targets, preds)
    pred_labels = (preds >= best_thresh).astype(int)

    print(f"\nBest Threshold: {best_thresh:.2f} - Best F1 Score: {best_f1:.4f}")
    print("Classification Report:")
    print(classification_report(targets, pred_labels, target_names=["Real", "Fake"]))
    print(f"Test AUC: {auc:.4f}")
    return auc

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

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    test_root = 'data/faces/test'  
    labels_dict = create_labels_dict(test_root)

    test_dataset = DeepfakeDataset(test_root, labels_dict, num_frames=20)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    model = DeepfakeModel().to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    print("Loaded best_model.pth")

    evaluate(model, test_loader, device)

if __name__ == '__main__':
    main()

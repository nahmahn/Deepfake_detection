import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from model import SVFTSwinFusion
from spectral_features import extract_spectral_features
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from tqdm import tqdm
import cv2

class VideoDataset(Dataset):
    def __init__(self, root_dir, labels_dict, num_frames=8):
        self.samples = list(labels_dict.items())
        self.root_dir = root_dir
        self.labels_dict = labels_dict
        self.num_frames = num_frames
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        full_path = os.path.join(self.root_dir, video_path)
        frame_files = sorted([
            os.path.join(full_path, f)
            for f in os.listdir(full_path) if f.lower().endswith(('.jpg', '.png'))
        ])[:self.num_frames]

        frame_preds = []
        for frame_file in frame_files:
            image = cv2.imread(frame_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            spectral = extract_spectral_features(image)
            spectral = torch.tensor(spectral, dtype=torch.float32)
            image = self.transform(image)
            frame_preds.append((image, spectral))

        images = torch.stack([img for img, _ in frame_preds])
        spectrals = torch.stack([sp for _, sp in frame_preds])

        return images, spectrals, torch.tensor(label, dtype=torch.float32)

def find_best_threshold(targets, preds):
    best_thresh, best_f1 = 0.5, 0
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

    loop = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for images, spectrals, labels in loop:
            b, f, c, h, w = images.shape
            images = images.view(-1, c, h, w).to(device)
            spectrals = spectrals.view(-1, spectrals.shape[-1]).to(device)

            outputs = model(images, spectrals)
            outputs = outputs.view(b, f, -1).mean(dim=1)  # average over frames
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu()
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
    for label in ['real', 'fake']:
        class_dir = os.path.join(root_dir, label)
        for video in os.listdir(class_dir):
            video_path = os.path.join(label, video)
            labels_dict[video_path] = 0 if label == 'real' else 1
    return labels_dict

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    test_root = 'frames/test'
    labels_dict = create_labels_dict(test_root)

    test_dataset = VideoDataset(test_root, labels_dict)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)

    model = SVFTSwinFusion().to(device)
    model.load_state_dict(torch.load("svft_swin_.pth", map_location=device))
    print("Loaded svft_swin_v1.pth")

    evaluate(model, test_loader, device)

if __name__ == '__main__':
    main()

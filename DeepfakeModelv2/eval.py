import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from model import DeepfakeBaseline
from dataset import DeepfakeDataset
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

test_root = r"C:\Users\namja\Deepfakev1\data\faces\test"
model_path = 'best_model.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_frames = 20
batch_size = 4

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  
])


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
            frames = frames.to(device)  # (B, T, C, H, W)
            labels = labels.to(device).float()
            outputs = model(frames)
            if outputs.dim() > 1:
                outputs = outputs.squeeze(1)
            probs = torch.sigmoid(outputs).cpu()
            preds.append(probs)
            targets.append(labels.cpu())

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    best_thresh, best_f1 = find_best_threshold(targets, preds)
    y_pred = (preds >= best_thresh).astype(int)

    print("\nClassification Report (Thresholded):")
    print(classification_report(targets, y_pred, target_names=["Real", "Fake"]))

    fpr, tpr, _ = roc_curve(targets, preds)
    roc_auc = auc(fpr, tpr)

    print(f"\nBest Threshold: {best_thresh:.3f}")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    cm = confusion_matrix(targets, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real', 'Fake'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Threshold = {best_thresh:.2f})")
    plt.show()

    return roc_auc

def create_labels_dict(root_dir):
    labels_dict = {}
    for label_name in ['real', 'fake']:
        class_dir = os.path.join(root_dir, label_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} does not exist.")
            continue
        for video_name in os.listdir(class_dir):
            video_path = os.path.join(label_name, video_name)
            full_video_path = os.path.join(root_dir, video_path)
            if os.path.isdir(full_video_path):
                labels_dict[video_path] = 0 if label_name == 'real' else 1
    return labels_dict

# --- Main ---
def main():
    print(f"Using device: {device}")
    labels_dict = create_labels_dict(test_root)

    if len(labels_dict) == 0:
        print("No videos found in test_root. Check your test dataset directory structure.")
        return

    test_dataset = DeepfakeDataset(test_root, labels_dict, num_frames=num_frames, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = DeepfakeBaseline().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loaded model weights")

    evaluate(model, test_loader, device)

if __name__ == '__main__':
    main()

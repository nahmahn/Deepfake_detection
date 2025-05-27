import os
import torch
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
from spectral_features import extract_spectral_features

class DeepfakeDataset(Dataset):
    def __init__(self, frame_root, transform=None):
        self.samples = []
        self.transform = transform
        for label in ['real', 'fake']:
            label_path = os.path.join(frame_root, label)
            for video_folder in os.listdir(label_path):
                frame_dir = os.path.join(label_path, video_folder)
                for frame in os.listdir(frame_dir):
                    self.samples.append((os.path.join(frame_dir, frame), 0 if label == 'real' else 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        freq_feat = extract_spectral_features(image)
        print("Spectral Features:", freq_feat)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(freq_feat, dtype=torch.float32), torch.tensor(label)


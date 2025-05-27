import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, labels_dict, num_frames=20, train=True):
        self.root_dir = root_dir
        self.labels_dict = labels_dict
        self.video_list = list(labels_dict.keys())
        self.num_frames = num_frames
        self.train = train

        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_key = self.video_list[idx]
        label = self.labels_dict[video_key]
        frames_dir = os.path.join(self.root_dir, video_key)
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])

        if len(frame_files) == 0:
            raise RuntimeError(f"No frames found in directory: {frames_dir}")

        if len(frame_files) < self.num_frames:
            frame_files += [frame_files[-1]] * (self.num_frames - len(frame_files))

        idxs = torch.linspace(0, len(frame_files) - 1, self.num_frames).long()
        selected = [frame_files[i] for i in idxs]

        transform = self.train_transform if self.train else self.val_transform

        frames = [transform(Image.open(os.path.join(frames_dir, f)).convert('RGB')) for f in selected]
        return torch.stack(frames), torch.tensor(label, dtype=torch.float)

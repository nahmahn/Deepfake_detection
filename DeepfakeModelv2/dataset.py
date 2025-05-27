import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DeepfakeDataset(Dataset):
    def __init__(self, faces_root, labels_dict, num_frames=10, transform=None):
        self.faces_root = faces_root
        self.labels = labels_dict
        self.num_frames = num_frames
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.video_list = list(self.labels.keys())

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_name = self.video_list[idx]
        frames_dir = os.path.join(self.faces_root, video_name)
        frame_files = sorted(os.listdir(frames_dir))

        if len(frame_files) < self.num_frames:
            frame_files = frame_files + [frame_files[-1]] * (self.num_frames - len(frame_files))
        else:
            frame_files = frame_files[:self.num_frames]

        frames = []
        for f in frame_files:
            img_path = os.path.join(frames_dir, f)
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            frames.append(img)

        frames = torch.stack(frames)  
        label = self.labels[video_name]

        return frames, torch.tensor(label, dtype=torch.float)


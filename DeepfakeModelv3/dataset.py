import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, labels_dict, num_frames=20):
        self.root_dir = root_dir
        self.labels_dict = labels_dict
        self.video_list = list(labels_dict.keys())
        self.num_frames = num_frames
        self.transform = transforms.Compose([
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
        frames = [self.transform(Image.open(os.path.join(frames_dir, f)).convert('RGB')) for f in selected]
        return torch.stack(frames), torch.tensor(label, dtype=torch.float)

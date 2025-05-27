import os
import cv2
import torch
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN
import sys
import numpy as np


sys.path.insert(0, 'C:/Users/namja/Deepfakev2/ESRGAN')
from RRDBNet_arch import RRDBNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

mtcnn = MTCNN(keep_all=False, device=device)

model_path = 'C:/Users/namja/Deepfakev2/ESRGAN/models/RRDB_ESRGAN_x4.pth'
sr_model = RRDBNet(3, 3, 64, 23, gc=32)
sr_model.load_state_dict(torch.load(model_path), strict=True)
sr_model.eval().to(device)

def enhance_image(img_pil):
    img = np.array(img_pil).astype(np.float32) / 255.0

    img = torch.from_numpy(np.transpose(img[:, :, ::-1].copy(), (2, 0, 1))).unsqueeze(0).to(device)

    with torch.no_grad():
        output = sr_model(img).squeeze().clamp_(0, 1).cpu().numpy()

    output = np.transpose(output, (1, 2, 0))[:, :, ::-1] 
    output = (output * 255.0).astype(np.uint8)

    return Image.fromarray(output)



def extract_faces(video_path, save_dir, num_frames=20):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f" {video_path} has 0 frames.")
        return

    frame_idxs = torch.linspace(0, total_frames - 1, steps=num_frames).long().tolist()
    saved = 0
    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(img_rgb)

        if boxes is not None:
            box = boxes[0].astype(int)
            x1, y1, x2, y2 = box
            face_crop = img_rgb[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            face_pil = Image.fromarray(face_crop) 
            face_sr = enhance_image(face_pil).resize((224, 224))
            face_sr.save(os.path.join(save_dir, f"frame_{saved:04d}.png"))
            saved += 1


    cap.release()

def preprocess_all(root_video_dir, root_save_dir, num_frames=20):
    for label in ['real', 'fake']:
        video_folder = os.path.join(root_video_dir, label)
        save_folder = os.path.join(root_save_dir, label)
        os.makedirs(save_folder, exist_ok=True)
        for file in tqdm(os.listdir(video_folder), desc=f"Processing {label}"):
            if not file.endswith(('.mp4', '.avi', '.mov')):
                continue
            name = os.path.splitext(file)[0]
            input_path = os.path.join(video_folder, file)
            output_path = os.path.join(save_folder, name)
            if not os.path.exists(output_path):
                extract_faces(input_path, output_path, num_frames=num_frames)

if __name__ == '__main__':
    preprocess_all('deepfake_dataset/train', 'data/faces/train', num_frames=20)
    preprocess_all('deepfake_dataset/test', 'data/faces/test', num_frames=20)

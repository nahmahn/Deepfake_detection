import os
import cv2
from facenet_pytorch import MTCNN
import torch
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, device=device)

def extract_faces_from_video(video_path, save_dir, num_frames=20):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Warning: {video_path} has 0 frames.")
        return

    frame_indices = torch.linspace(0, total_frames - 1, steps=num_frames).long().tolist()

    saved = 0
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = mtcnn(img_rgb)
        if face is not None:
            face_img = face.permute(1, 2, 0).clamp(-1, 1) 
            face_img = ((face_img + 1) / 2 * 255).byte().cpu().numpy()  
            cv2.imwrite(os.path.join(save_dir, f'frame_{saved:04d}.jpg'), cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))

            saved += 1

    cap.release()


def preprocess_all_videos(root_videos_dir, root_save_dir, num_frames=10):
    for label_folder in ['real', 'fake']:
        videos_path = os.path.join(root_videos_dir, label_folder)
        save_path = os.path.join(root_save_dir, label_folder)
        os.makedirs(save_path, exist_ok=True)

        for video_file in tqdm(os.listdir(videos_path), desc=f"Processing {label_folder} videos"):
            video_name = os.path.splitext(video_file)[0]
            video_path = os.path.join(videos_path, video_file)
            video_save_dir = os.path.join(save_path, video_name)
            if not os.path.exists(video_save_dir):  
                extract_faces_from_video(video_path, video_save_dir, num_frames=num_frames)

if __name__ == '__main__':
    preprocess_all_videos('deepfake_dataset/train', 'data/faces/train', num_frames=20)
    preprocess_all_videos('deepfake_dataset/test', 'data/faces/test', num_frames=20)

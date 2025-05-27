import os
import cv2

def extract_frames(src_dir, dst_dir, every_n=10, max_frames=8):
    os.makedirs(dst_dir, exist_ok=True)
    for label in ['real', 'fake']:
        in_path = os.path.join(src_dir, label)
        out_path = os.path.join(dst_dir, label)
        os.makedirs(out_path, exist_ok=True)

        for video in os.listdir(in_path):
            cap = cv2.VideoCapture(os.path.join(in_path, video))
            video_name = os.path.splitext(video)[0]
            save_dir = os.path.join(out_path, video_name)
            os.makedirs(save_dir, exist_ok=True)

            frame_id = 0
            saved = 0
            while cap.isOpened() and saved < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_id % every_n == 0:
                    frame_path = os.path.join(save_dir, f"frame_{saved:04d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    saved += 1

                frame_id += 1
            cap.release()




extract_frames("deepfake_dataset/train", "frames/train")
extract_frames("deepfake_dataset/test", "frames/test")
# Deepfake Detection

This project is a deep learning pipeline for detecting deepfake videos using PyTorch. It includes preprocessing, training, and evaluation scripts, and leverages a ResNeXt backbone with LSTM for temporal modeling.<br />
[Dataset Link](https://drive.google.com/file/d/1DKOkBIAy7HyP91N34mj1E0FFu7sqdKR_/view)<br />
[Download the .pth file for the Model4 from here](https://drive.google.com/file/d/1qz-RxRYeFx-lbhIa9Ht8VWYjYLD6uaA5/view?usp=drive_link)

## Features
- Face extraction and super-resolution preprocessing
- Custom PyTorch dataset and dataloader
- ResNeXt50 backbone with LSTM for sequence modeling
- Training with early stopping and backbone fine-tuning
- Evaluation with ROC, F1, and confusion matrix visualization

## Directory Structure
```
DeepfakeModelv4/
├──ERSGAN/(git clone)
├──deepfake_dataset
├──data
   ├──faces
      ├──test
      ├──train
├── best_model.pth         # Trained model weights
├── dataset.py             # Custom dataset class
├── eval.py                # Evaluation script
├── model.py               # Model definition
├── preprocess.py          # Preprocessing (face extraction, super-resolution)
├── train.py               # Training script          
└── __init__.py            # Package marker
```

## Requirements
- Python 3.7+
- PyTorch
- torchvision
- scikit-learn
- tqdm
- numpy
- opencv-python
- pillow
- facenet-pytorch
- seaborn
- matplotlib

You may install dependencies with:
```bash
pip install torch torchvision scikit-learn tqdm numpy opencv-python pillow facenet-pytorch seaborn matplotlib
```

## Data Preparation
1. Place your raw videos in `deepfake_dataset/train/real`, `deepfake_dataset/train/fake`, `deepfake_dataset/test/real`, and `deepfake_dataset/test/fake`.
2. Run the preprocessing script to extract faces and apply super-resolution:
   ```bash
   python preprocess.py
   ```
   This will create processed frames in `data/faces/train` and `data/faces/test`.

## Training
Train the model using:
```bash
python train.py
```
The best model will be saved as `best_model.pth`.

## Evaluation
Evaluate the trained model:
```bash
python eval.py
```
This will print metrics and show confusion matrix and ROC curve plots.

These are the comparison with other model architectures trained on the same dataset

![Screenshot 2025-05-28 011025](https://github.com/user-attachments/assets/034b07fb-3fbd-48cd-a9cd-662cad39e0e6)

## Notes
- The preprocessing script uses ESRGAN for super-resolution. Make sure the ESRGAN model weights and code are available as referenced in `preprocess.py`.
- Adjust paths in scripts as needed for your environment.

## License
MIT License

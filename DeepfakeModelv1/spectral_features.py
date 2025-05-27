import numpy as np
import cv2
from scipy.stats import pearsonr

def extract_spectral_features(image):
    channels = cv2.split(image)
    specs = [np.abs(np.fft.fft2(ch)) for ch in channels]

    diff_rg = np.abs(specs[0] - specs[1])
    diff_rb = np.abs(specs[0] - specs[2])
    diff_gb = np.abs(specs[1] - specs[2])

    mean_rg = np.mean(diff_rg)
    mean_rb = np.mean(diff_rb)
    mean_gb = np.mean(diff_gb)

    mean_avg = np.mean([mean_rg, mean_rb, mean_gb])
    min_avg = np.min([mean_rg, mean_rb, mean_gb])
    max_avg = np.max([mean_rg, mean_rb, mean_gb])

    corr_rg = pearsonr(specs[0].flatten(), specs[1].flatten())[0]
    corr_rb = pearsonr(specs[0].flatten(), specs[2].flatten())[0]
    corr_gb = pearsonr(specs[1].flatten(), specs[2].flatten())[0]

    return np.array([mean_rg, mean_rb, mean_gb, mean_avg, min_avg, max_avg, corr_rg, corr_rb, corr_gb], dtype=np.float32)

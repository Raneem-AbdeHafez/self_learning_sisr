import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def resize_to_match(ref, img):
    return cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_CUBIC)

def enhance_contrast_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def compute_metrics(img1, img2):
    psnr_val = psnr(img1, img2, data_range=255)
    ssim_val = ssim(img1, img2, channel_axis=2, data_range=255)
    return psnr_val, ssim_val

# === Load Images ===
sr_path = 'refined_image_color_minibatch.jpg'
hr_path = 'images/high_res.jpg'
lr_path = 'images/low_res_color.jpg'

sr = cv2.imread(sr_path)
hr = cv2.imread(hr_path)
lr = cv2.imread(lr_path)

if sr is None or hr is None or lr is None:
    raise FileNotFoundError("Make sure all image paths are correct and files exist.")

# Resize all to match HR dimensions
sr = resize_to_match(hr, sr)
lr = resize_to_match(hr, lr)

# === Post-processing pipeline (without sharpening) ===
denoised = cv2.bilateralFilter(sr, d=9, sigmaColor=75, sigmaSpace=75)
final_sr = enhance_contrast_clahe(denoised)

# === Compute Metrics ===
psnr_hr, ssim_hr = compute_metrics(final_sr, hr)
psnr_sr, ssim_sr = compute_metrics(final_sr, sr)

# === Output ===
print(f"\nPost-processed SR Quality:")
print(f"PSNR vs HR: {psnr_hr:.2f} dB")
print(f"SSIM vs HR: {ssim_hr:.4f}")
print(f"PSNR vs Original SR: {psnr_sr:.2f} dB")
print(f"SSIM vs Original SR: {ssim_sr:.4f}")

# Save final result
cv2.imwrite("images/final_sr_processed.jpg", final_sr)
print("\n💾 Final image saved as 'images/final_sr_processed.jpg'")

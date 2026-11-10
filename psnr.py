import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

  #  super-resolved or upsampled image
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

def load_and_prepare(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def resize_to_match(source, target_shape):
    return cv2.resize(source, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)

def calculate_psnr(img1, img2):
    psnr_r = psnr(img1[:, :, 0], img2[:, :, 0], data_range=255)
    psnr_g = psnr(img1[:, :, 1], img2[:, :, 1], data_range=255)
    psnr_b = psnr(img1[:, :, 2], img2[:, :, 2], data_range=255)
    psnr_avg = np.mean([psnr_r, psnr_g, psnr_b])
    return psnr_r, psnr_g, psnr_b, psnr_avg

# === File paths ===
hr_path = 'images/high_res.jpg'
lr_path = 'images/low_res_color.jpg'
sr_path = 'images/refined_image_color_minibatch.jpg'

# === Load and prepare ===
hr = load_and_prepare(hr_path)
lr = load_and_prepare(lr_path)
sr = load_and_prepare(sr_path)

# === Resize LR and SR to match HR (if needed) ===
if hr.shape != lr.shape:
    print("Resizing LR image to match HR...")
    lr = resize_to_match(lr, hr.shape)

if hr.shape != sr.shape:
    print("Resizing SR image to match HR...")
    sr = resize_to_match(sr, hr.shape)

# === Compare SR vs HR ===
psnr_r1, psnr_g1, psnr_b1, psnr_avg1 = calculate_psnr(hr, sr)
print("\n🔍 PSNR: Super-Resolved vs High-Res")
print(f"  Red:   {psnr_r1:.2f} dB")
print(f"  Green: {psnr_g1:.2f} dB")
print(f"  Blue:  {psnr_b1:.2f} dB")
print(f"  Avg:   {psnr_avg1:.2f} dB")

# === Compare SR vs LR ===
psnr_r2, psnr_g2, psnr_b2, psnr_avg2 = calculate_psnr(lr, sr)
print("\n🔍 PSNR: Super-Resolved vs Low-Res")
print(f"  Red:   {psnr_r2:.2f} dB")
print(f"  Green: {psnr_g2:.2f} dB")
print(f"  Blue:  {psnr_b2:.2f} dB")
print(f"  Avg:   {psnr_avg2:.2f} dB")

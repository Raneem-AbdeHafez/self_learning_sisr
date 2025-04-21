import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

# === File paths ===
path_sr = "refined_image_color_minibatch.jpg"   # Output image from your model
path_lr = "images/low_res_color.jpg"    # Original low-res image
path_hr = "images/high_res.jpg"  # Ground truth high-res image

# === Load images ===
sr_rgb = cv2.cvtColor(cv2.imread(path_sr), cv2.COLOR_BGR2RGB)
lr_rgb = cv2.cvtColor(cv2.imread(path_lr), cv2.COLOR_BGR2RGB)
hr_rgb = cv2.cvtColor(cv2.imread(path_hr), cv2.COLOR_BGR2RGB)

# === Convert LR to YCbCr ===
lr_ycbcr = cv2.cvtColor(lr_rgb, cv2.COLOR_RGB2YCrCb)
y_lr, cb_lr, cr_lr = cv2.split(lr_ycbcr)

# === Use SR luminance (Y) ===
y_sr = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2GRAY)

# === Upscale Cb and Cr to SR size ===
cb_up = cv2.resize(cb_lr, (y_sr.shape[1], y_sr.shape[0]), interpolation=cv2.INTER_CUBIC)
cr_up = cv2.resize(cr_lr, (y_sr.shape[1], y_sr.shape[0]), interpolation=cv2.INTER_CUBIC)

# === Merge Y + upsampled CbCr ===
merged_ycrcb = cv2.merge([y_sr, cb_up, cr_up])
sr_final_rgb = cv2.cvtColor(merged_ycrcb, cv2.COLOR_YCrCb2RGB)

# === Compute PSNR vs HR ===
def compute_psnr_metrics(img1, img2):
    r_psnr = psnr(img1[..., 0], img2[..., 0])
    g_psnr = psnr(img1[..., 1], img2[..., 1])
    b_psnr = psnr(img1[..., 2], img2[..., 2])
    avg = (r_psnr + g_psnr + b_psnr) / 3
    return r_psnr, g_psnr, b_psnr, avg

# Resize HR image to match SR final output (if needed)
if sr_final_rgb.shape != hr_rgb.shape:
    hr_rgb = cv2.resize(hr_rgb, (sr_final_rgb.shape[1], sr_final_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)

r, g, b, avg = compute_psnr_metrics(sr_final_rgb, hr_rgb)

# === Print new PSNRs ===
print("📈 PSNR: Final SR vs High-Res")
print(f"  Red:   {r:.2f} dB")
print(f"  Green: {g:.2f} dB")
print(f"  Blue:  {b:.2f} dB")
print(f"  Avg:   {avg:.2f} dB")

# Optional: Save the post-processed image
cv2.imwrite("sr_postprocessed.png", cv2.cvtColor(sr_final_rgb, cv2.COLOR_RGB2BGR))



from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.color import rgb2ycbcr

sr_y = rgb2ycbcr(sr_final_rgb)[..., 0]
hr_y = rgb2ycbcr(hr_rgb)[..., 0]

y_psnr = psnr(hr_y, sr_y, data_range=255)
print(f"PSNR (Y channel only): {y_psnr:.2f} dB")

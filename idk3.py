
#VERY SLOW
# his code implements a self-learning approach similar to the one described in the paper. 
# here is a small difference: this code uses Ridge regression instead of the Support Vector Regression (SVR) used in the paper. 
"""import numpy as np
import cv2
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.decomposition import DictionaryLearning
from skimage.util import view_as_windows
import os
import time

# --- PARAMETERS ---
PATCH_SIZE = 5
STRIDE = 2                  # Increased to reduce number of patches
DICT_COMPONENTS = 50        # Reduced dictionary size
SPARSITY_ALPHA = 1.0
UPSCALE = 2
MAX_PATCHES = 10000         # Number of training samples for speed

# --- PATCH UTILS ---
def extract_patches(img, patch_size, stride):
    patches = view_as_windows(img, (patch_size, patch_size), step=stride)
    h, w, _, _ = patches.shape
    return patches.reshape(h * w, patch_size * patch_size), (h, w)

def reconstruct_image(patches, img_shape, patch_size, stride):
    h_patches, w_patches = img_shape
    rec = np.zeros((h_patches * stride + patch_size - stride,
                    w_patches * stride + patch_size - stride))
    weight = np.zeros_like(rec)

    idx = 0
    for i in range(0, h_patches * stride, stride):
        for j in range(0, w_patches * stride, stride):
            patch = patches[idx].reshape(patch_size, patch_size)
            rec[i:i+patch_size, j:j+patch_size] += patch
            weight[i:i+patch_size, j:j+patch_size] += 1
            idx += 1

    return (rec / weight).clip(0, 255).astype(np.uint8)

# --- MAIN ---
start_time = time.time()

# Load and optionally downscale image
hr_image = Image.open("images/high_res.jpg").convert("L")
hr_y = np.array(hr_image)
print("Original HR image shape:", hr_y.shape)

# Downscale HR image to create LR version
lr_y = cv2.resize(hr_y, (hr_y.shape[1] // UPSCALE, hr_y.shape[0] // UPSCALE), interpolation=cv2.INTER_CUBIC)
lr_up = cv2.resize(lr_y, (hr_y.shape[1], hr_y.shape[0]), interpolation=cv2.INTER_CUBIC)

# Extract patches
print("Extracting patches...")
lr_patches, shape = extract_patches(lr_up, PATCH_SIZE, STRIDE)
hr_patches, _ = extract_patches(hr_y, PATCH_SIZE, STRIDE)
residuals = hr_patches - lr_patches

# Randomly sample patches for training
print("Sampling patches...")
np.random.seed(42)
indices = np.random.choice(len(lr_patches), size=min(MAX_PATCHES, len(lr_patches)), replace=False)
lr_patches_train = lr_patches[indices]
residuals_train = residuals[indices]

# Dictionary learning
print("Learning dictionary...")
dict_learner = DictionaryLearning(n_components=DICT_COMPONENTS, alpha=SPARSITY_ALPHA, transform_algorithm='omp')
dict_learner.fit(lr_patches_train)

# Transform patches to sparse codes
print("Transforming training patches...")
sparse_codes_train = dict_learner.transform(lr_patches_train)

# Train regressor (Ridge instead of SVR)
print("Training ridge regression...")
ridge = Ridge(alpha=1.0)
ridge.fit(sparse_codes_train, residuals_train)

# Predict on all patches
print("Transforming all patches...")
sparse_codes_all = dict_learner.transform(lr_patches)
predicted_residuals = ridge.predict(sparse_codes_all)
final_patches = lr_patches + predicted_residuals

# Reconstruct image
print("Reconstructing final image...")
final_hr = reconstruct_image(final_patches, shape, PATCH_SIZE, STRIDE)

# Save output
os.makedirs("output", exist_ok=True)
Image.fromarray(final_hr).save("output/sisr_result.jpg")

# Done
print("SISR completed in {:.2f} seconds.".format(time.time() - start_time))
print("Output saved to output/sisr_result.jpg")
"""



# FASTER, no color output and no evalutation
# this code is close to the paper's approach, but it uses a slightly more efficient dictionary learning method (MiniBatchDictionaryLearning) and a different regression model (Ridge vs SVR).
"""
import numpy as np
import cv2
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.decomposition import MiniBatchDictionaryLearning
from skimage.util import view_as_windows
import os
import time

# --- PARAMETERS ---
PATCH_SIZE = 5
STRIDE = 2
DICT_COMPONENTS = 50
SPARSITY_ALPHA = 1.0
UPSCALE = 2
MAX_PATCHES = 10000

# --- PATCH UTILS ---
def extract_patches(img, patch_size, stride):
    patches = view_as_windows(img, (patch_size, patch_size), step=stride)
    h, w, _, _ = patches.shape
    return patches.reshape(h * w, patch_size * patch_size), (h, w)

def reconstruct_image(patches, img_shape, patch_size, stride):
    h_patches, w_patches = img_shape
    rec = np.zeros((h_patches * stride + patch_size - stride,
                    w_patches * stride + patch_size - stride))
    weight = np.zeros_like(rec)

    idx = 0
    for i in range(0, h_patches * stride, stride):
        for j in range(0, w_patches * stride, stride):
            patch = patches[idx].reshape(patch_size, patch_size)
            rec[i:i+patch_size, j:j+patch_size] += patch
            weight[i:i+patch_size, j:j+patch_size] += 1
            idx += 1

    return (rec / weight).clip(0, 255).astype(np.uint8)

# --- MAIN ---
start_time = time.time()

# Load and optionally downscale image
hr_image = Image.open("images/high_res.jpg").convert("L")
hr_y = np.array(hr_image)
print("Original HR image shape:", hr_y.shape)

# Downscale HR image to create LR version
lr_y = cv2.resize(hr_y, (hr_y.shape[1] // UPSCALE, hr_y.shape[0] // UPSCALE), interpolation=cv2.INTER_CUBIC)
lr_up = cv2.resize(lr_y, (hr_y.shape[1], hr_y.shape[0]), interpolation=cv2.INTER_CUBIC)

# Extract patches
print("Extracting patches...")
lr_patches, shape = extract_patches(lr_up, PATCH_SIZE, STRIDE)
hr_patches, _ = extract_patches(hr_y, PATCH_SIZE, STRIDE)
residuals = hr_patches - lr_patches

# Randomly sample patches for training
print("Sampling patches...")
np.random.seed(42)
indices = np.random.choice(len(lr_patches), size=min(MAX_PATCHES, len(lr_patches)), replace=False)
lr_patches_train = lr_patches[indices]
residuals_train = residuals[indices]

# Dictionary learning (faster with MiniBatch)
print("Learning dictionary...")
dict_learner = MiniBatchDictionaryLearning(
    n_components=DICT_COMPONENTS,
    alpha=SPARSITY_ALPHA,
    transform_algorithm='lasso_lars',
    n_iter=100,
    batch_size=256,
    verbose=True
)
dict_learner.fit(lr_patches_train)

# Transform patches to sparse codes
print("Transforming training patches...")
sparse_codes_train = dict_learner.transform(lr_patches_train)

# Train regressor (Ridge instead of SVR)
print("Training ridge regression...")
ridge = Ridge(alpha=1.0)
ridge.fit(sparse_codes_train, residuals_train)

# Predict on all patches
print("Transforming all patches...")
sparse_codes_all = dict_learner.transform(lr_patches)
predicted_residuals = ridge.predict(sparse_codes_all)
final_patches = lr_patches + predicted_residuals

# Reconstruct image
print("Reconstructing final image...")
final_hr = reconstruct_image(final_patches, shape, PATCH_SIZE, STRIDE)

# Save output
os.makedirs("output", exist_ok=True)
Image.fromarray(final_hr).save("output/sisr_result.jpg")

# Done
print("SISR completed in {:.2f} seconds.".format(time.time() - start_time))
print("Output saved to output/sisr_result.jpg")
"""



# VERY FAST, colored image but it cubes
"""
import numpy as np
from skimage.io import imread, imsave
from skimage.util import view_as_blocks
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
import time
from tqdm import tqdm

# === CONFIG ===
PATCH_SIZE = 32
STRIDE = 32
LOW_RES_PATH = 'images/input.jpg'
HR_GROUND_TRUTH_PATH = 'images/high_res.jpg'
OUTPUT_IMAGE_PATH = 'output/sisr_result.jpg'

# === TIMER START ===
start_time = time.time()

# === STEP 1: LOAD IMAGES ===
lr_img = imread(LOW_RES_PATH)
hr_rgb = imread(HR_GROUND_TRUTH_PATH)

# Convert to uint8 if needed
if lr_img.dtype != np.uint8:
    lr_img = (lr_img * 255).astype(np.uint8)
if hr_rgb.dtype != np.uint8:
    hr_rgb = (hr_rgb * 255).astype(np.uint8)

# === STEP 2: PATCHIFY LOW-RES IMAGE ===
def extract_patches(image, patch_size, stride):
    patches = []
    positions = []
    h, w, c = image.shape
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
            positions.append((y, x))
    return np.array(patches), positions

# === STEP 3: DUMMY TRANSFORM (REPLACE WITH MODEL) ===
def transform_patch(patch):
    upscale_factor = 2
    return resize(patch, (patch.shape[0]*upscale_factor, patch.shape[1]*upscale_factor), anti_aliasing=True)

def process_patches(patches):
    print("Transforming all patches...")
    transformed_patches = []
    for patch in tqdm(patches):
        transformed = transform_patch(patch)
        transformed = np.clip(transformed * 255, 0, 255).astype(np.uint8)
        transformed_patches.append(transformed)
    return transformed_patches

# === STEP 4: RECONSTRUCT FINAL IMAGE ===
def reconstruct_image(patches, positions, out_shape):
    print("Reconstructing final image...")
    recon_img = np.zeros(out_shape, dtype=np.uint16)  # Use uint16 to avoid overflow
    counts = np.zeros(out_shape[:2], dtype=np.uint8)

    for patch, (y, x) in zip(patches, positions):
        recon_img[y:y+patch.shape[0], x:x+patch.shape[1]] += patch
        counts[y:y+patch.shape[0], x:x+patch.shape[1]] += 1

    counts = np.maximum(counts, 1)[..., np.newaxis]
    final_img = (recon_img / counts).astype(np.uint8)
    return final_img

# === MAIN PIPELINE ===
patches, positions = extract_patches(lr_img, PATCH_SIZE, STRIDE)
transformed_patches = process_patches(patches)

upscale = transformed_patches[0].shape[0] // PATCH_SIZE
new_h = lr_img.shape[0] * upscale
new_w = lr_img.shape[1] * upscale
out_shape = (new_h, new_w, 3)
scaled_positions = [(y*upscale, x*upscale) for (y, x) in positions]

final_hr = reconstruct_image(transformed_patches, scaled_positions, out_shape)
imsave(OUTPUT_IMAGE_PATH, final_hr)

# === STEP 5: UPSCALE LR FOR COMPARISON ===
lr_up_rgb = resize(lr_img, (new_h, new_w), preserve_range=True, anti_aliasing=True).astype(np.uint8)

# Resize HR ground truth if needed
if hr_rgb.shape != final_hr.shape:
    hr_rgb = resize(hr_rgb, final_hr.shape, preserve_range=True, anti_aliasing=True).astype(np.uint8)

# === STEP 6: COMPUTE METRICS ===
print("Computing PSNR and SSIM...")
psnr_hr_sr = psnr(hr_rgb, final_hr)
ssim_hr_sr = ssim(hr_rgb, final_hr, channel_axis=-1, win_size=7)

psnr_lr_sr = psnr(lr_up_rgb, final_hr)
ssim_lr_sr = ssim(lr_up_rgb, final_hr, channel_axis=-1, win_size=7)

# === DONE ===
print("SISR completed in {:.2f} seconds.".format(time.time() - start_time))
print(f"Output saved to {OUTPUT_IMAGE_PATH}")
print(f"PSNR between HR and SR output: {psnr_hr_sr:.2f} dB")
print(f"SSIM between HR and SR output: {ssim_hr_sr:.4f}")
print(f"PSNR between LR and SR output: {psnr_lr_sr:.2f} dB")
print(f"SSIM between LR and SR output: {ssim_lr_sr:.4f}")
"""



# This code is structured well for a basic super-resolution pipeline with patch-based transformations.
# This code is a good start for implementing a simple image transformation and super-resolution pipeline, but it falls short of the more advanced techniques used in the latest papers.
"""
import numpy as np
from PIL import Image
import os
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import cv2

# --------- Helpers ----------
def load_image(path):
    return np.array(Image.open(path))

def save_image(img_array, path):
    img = Image.fromarray(np.uint8(img_array))
    img.save(path)

def rgb2y(img):
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

def patchify(img, patch_size):
    patches = []
    h, w = img.shape[:2]
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append((patch, (i, j)))
    return patches

def unpatchify(patches, img_shape, patch_size):
    h, w = img_shape[:2]
    channels = patches[0][0].shape[2] if patches[0][0].ndim == 3 else 1
    final_img = np.zeros((h, w, channels), dtype=np.float32)

    for patch, (i, j) in patches:
        final_img[i:i+patch_size, j:j+patch_size] = patch

    return final_img.astype(np.uint8)

def transform_patch(patch):
    # Sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    
    # Apply filter to each channel
    sharpened = np.zeros_like(patch)
    for i in range(3):  # for RGB channels
        sharpened[..., i] = cv2.filter2D(patch[..., i], -1, kernel)
    
    # Clip to valid range
    return np.clip(sharpened, 0, 255).astype(np.uint8)

# --------- Main ----------
start_time = time.time()

# Load HR and LR images
hr_rgb = load_image("images/high_res.jpg")            # ground truth
lr_rgb = load_image("images/input.jpg")         # input low-res

# Upsample LR to match HR dimensions
lr_up_rgb = np.array(Image.fromarray(lr_rgb).resize(hr_rgb.shape[1::-1], Image.BICUBIC))

# Process image patches (sharpening transformation)
patch_size = 8
sr_patches = []

print("Transforming all patches...")
patches = patchify(lr_up_rgb, patch_size)

for patch, coord in tqdm(patches):
    # Apply sharpening transformation here
    transformed = transform_patch(patch)
    sr_patches.append((transformed, coord))

# Reconstruct full SR image
print("Reconstructing final image...")
final_hr = unpatchify(sr_patches, hr_rgb.shape, patch_size)

# If needed, clip values
final_hr = np.clip(final_hr, 0, 255).astype(np.uint8)

# Save output
os.makedirs("output", exist_ok=True)
save_image(final_hr, "output/sisr_result.jpg")

# ---------- Metrics ----------
print("Computing PSNR and SSIM...")

# Ensure size match
assert hr_rgb.shape == final_hr.shape == lr_up_rgb.shape

# Handle new skimage versions with channel_axis
psnr_hr_sr = psnr(hr_rgb, final_hr, data_range=255)
ssim_hr_sr = ssim(hr_rgb, final_hr, channel_axis=-1, data_range=255)

psnr_lr_sr = psnr(lr_up_rgb, final_hr, data_range=255)
ssim_lr_sr = ssim(lr_up_rgb, final_hr, channel_axis=-1, data_range=255)

# ---------- Done ----------
print("SISR completed in {:.2f} seconds.".format(time.time() - start_time))
print("Output saved to output/sisr_result.jpg")
print(f"PSNR between HR and SR output: {psnr_hr_sr:.2f} dB")
print(f"SSIM between HR and SR output: {ssim_hr_sr:.4f}")
print(f"PSNR between LR and SR output: {psnr_lr_sr:.2f} dB")
print(f"SSIM between LR and SR output: {ssim_lr_sr:.4f}")
"""



#Code to Match the Paper’s, not really.
# In terms of structure and methodology, it looks like this code follows an approach similar to traditional image super-resolution methods but may not be identical to what some specific papers might use
"""
import numpy as np
import cv2
from PIL import Image
from sklearn.svm import SVR
from sklearn.decomposition import DictionaryLearning
from skimage.util import view_as_windows
import os
import time

# --- PARAMETERS ---
PATCH_SIZE = 5
STRIDE = 2
DICT_COMPONENTS = 64   # Number of dictionary components
SPARSITY_ALPHA = 1.0
UPSCALE = 2            # Factor to upscale the LR image to HR
MAX_PATCHES = 10000    # Number of patches for training

# --- PATCH UTILS ---
def extract_patches(img, patch_size, stride):
    patches = view_as_windows(img, (patch_size, patch_size), step=stride)
    h, w, _, _ = patches.shape
    return patches.reshape(h * w, patch_size * patch_size), (h, w)

def reconstruct_image(patches, img_shape, patch_size, stride):
    h_patches, w_patches = img_shape
    rec = np.zeros((h_patches * stride + patch_size - stride,
                    w_patches * stride + patch_size - stride))
    weight = np.zeros_like(rec)

    idx = 0
    for i in range(0, h_patches * stride, stride):
        for j in range(0, w_patches * stride, stride):
            patch = patches[idx].reshape(patch_size, patch_size)
            rec[i:i+patch_size, j:j+patch_size] += patch
            weight[i:i+patch_size, j:j+patch_size] += 1
            idx += 1

    return (rec / weight).clip(0, 255).astype(np.uint8)

# --- EVALUATION METRICS ---
def calculate_mse(original, predicted):
    return np.mean((original - predicted) ** 2)

def calculate_psnr(original, predicted):
    mse = calculate_mse(original, predicted)
    if mse == 0:
        return 100  # Max PSNR for identical images
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# --- MAIN ---
start_time = time.time()

# Load high-resolution image and convert to grayscale
hr_image = Image.open("images/high_res.jpg").convert("L")
hr_y = np.array(hr_image)
print("Original HR image shape:", hr_y.shape)

# Downscale HR image to create LR version
lr_y = cv2.resize(hr_y, (hr_y.shape[1] // UPSCALE, hr_y.shape[0] // UPSCALE), interpolation=cv2.INTER_CUBIC)
lr_up = cv2.resize(lr_y, (hr_y.shape[1], hr_y.shape[0]), interpolation=cv2.INTER_CUBIC)

# Extract patches
print("Extracting patches...")
lr_patches, shape = extract_patches(lr_up, PATCH_SIZE, STRIDE)
hr_patches, _ = extract_patches(hr_y, PATCH_SIZE, STRIDE)

# Dictionary learning (learn dictionary from LR patches)
print("Learning dictionary...")
dict_learner = DictionaryLearning(n_components=DICT_COMPONENTS, alpha=SPARSITY_ALPHA, fit_algorithm='lars', max_iter=50)
dict_learner.fit(lr_patches)

# Transform patches to sparse codes
print("Transforming patches to sparse codes...")
sparse_codes_lr = dict_learner.transform(lr_patches)

# Train Support Vector Regression (SVR) model
print("Training SVR model...")
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr.fit(sparse_codes_lr, hr_patches)

# Predict high-resolution patches from low-resolution patches
print("Predicting high-resolution patches...")
predicted_hr_patches = svr.predict(sparse_codes_lr)

# Reconstruct the final high-resolution image
print("Reconstructing final HR image...")
final_hr = reconstruct_image(predicted_hr_patches, shape, PATCH_SIZE, STRIDE)

# Save output
os.makedirs("output", exist_ok=True)
Image.fromarray(final_hr).save("output/sisr_result.jpg")

# Evaluation
mse = calculate_mse(hr_y, final_hr)
psnr = calculate_psnr(hr_y, final_hr)

# Done
print("SISR completed in {:.2f} seconds.".format(time.time() - start_time))
print("Output saved to output/sisr_result.jpg")

# Print evaluation metrics
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr:.4f} dB")

"""

# simplified implementation of the framework described in the paper
"""
import cv2
import numpy as np
from sklearn.decomposition import DictionaryLearning
from sklearn.svm import SVR

# Function to upsample image using bicubic interpolation
def upsample_image(image, scale_factor):
    height, width = image.shape[:2]
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

# Function to extract patches from an image
def extract_patches(image, patch_size, stride):
    patches = []
    height, width = image.shape[:2]
    
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch.flatten())  # Flatten the patch to a vector
    
    return np.array(patches)

# Dictionary learning for sparse representation
def learn_dictionary(patches, n_components=100):
    dictionary = DictionaryLearning(n_components=n_components)
    dictionary.fit(patches)
    return dictionary

# Train SVR model
def train_svr(patches, target_patches):
    svr_model = SVR(kernel='rbf', C=1, epsilon=0.1)
    svr_model.fit(patches, target_patches)
    return svr_model

# Predict residuals using SVR
def predict_residuals(svr_model, patches):
    return svr_model.predict(patches)

# Refine the upsampled image
def refine_image(upsampled_image, patches, residuals, stride):
    height, width = upsampled_image.shape[:2]
    refined_image = upsampled_image.copy()
    
    idx = 0
    for y in range(0, height - patches.shape[1] + 1, stride):
        for x in range(0, width - patches.shape[2] + 1, stride):
            patch = refined_image[y:y + patches.shape[1], x:x + patches.shape[2]]
            residual_patch = residuals[idx].reshape(patch.shape)
            refined_image[y:y + patches.shape[1], x:x + patches.shape[2]] = patch + residual_patch
            idx += 1
            
    return refined_image

# Full SR pipeline
def super_resolve_image(image, scale_factor=2, patch_size=8, stride=6):
    upsampled_image = upsample_image(image, scale_factor)
    patches = extract_patches(upsampled_image, patch_size, stride)
    dictionary = learn_dictionary(patches)
    residuals = predict_residuals(dictionary, patches)
    refined_image = refine_image(upsampled_image, patches, residuals, stride)
    return refined_image

# Load and process image
image = cv2.imread('low_res_image.jpg', cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
refined_image = super_resolve_image(image)

# Save the result
cv2.imwrite('refined_image.jpg', refined_image)


"""



# This implementation follows the framework outlined in the paper, using both dictionary learning and SVR for residual prediction. 
# REALLY REALLY SLOW

"""import cv2
import numpy as np
from sklearn.decomposition import DictionaryLearning
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 1. Upsampling using Bicubic Interpolation
def upsample_image(image, scale_factor):
    height, width = image.shape[:2]
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

# 2. Patch Extraction
def extract_patches(image, patch_size, stride):
    patches = []
    positions = []
    height, width = image.shape[:2]
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch.flatten())
            positions.append((y, x))
    return np.array(patches), positions

# 3. Patch Aggregation
def aggregate_patches(image_shape, patch_size, stride, patches, positions):
    reconstructed = np.zeros(image_shape, dtype=np.float32)
    weight_map = np.zeros(image_shape, dtype=np.float32)

    idx = 0
    for (y, x) in positions:
        patch = patches[idx].reshape((patch_size, patch_size))
        reconstructed[y:y + patch_size, x:x + patch_size] += patch
        weight_map[y:y + patch_size, x:x + patch_size] += 1
        idx += 1

    # Avoid division by zero
    weight_map[weight_map == 0] = 1
    return (reconstructed / weight_map).astype(np.uint8)

# 4. Sparse Representation using Dictionary Learning
def learn_dictionary(patches, n_components=100):
    dict_learner = DictionaryLearning(n_components=n_components, fit_algorithm='lars', transform_algorithm='lasso_lars', random_state=0)
    dict_learner.fit(patches)
    return dict_learner

# 5. Support Vector Regression (SVR)
def train_svr(X, y):
    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.01)
    svr_model.fit(X, y)
    return svr_model

def predict_residuals(svr_model, sparse_repr):
    return svr_model.predict(sparse_repr)

# 6. Iterative Refinement
def iterative_refinement(image, scale_factor=2, patch_size=8, stride=4, n_iterations=3, n_components=100):
    upsampled = upsample_image(image, scale_factor)
    upsampled = upsampled.astype(np.float32)

    for iter_num in range(n_iterations):
        print(f"Iteration {iter_num + 1}/{n_iterations}...")

        # Extract patches
        patches, positions = extract_patches(upsampled, patch_size, stride)

        # Normalize patches
        scaler = StandardScaler()
        patches_normalized = scaler.fit_transform(patches)

        # Learn dictionary
        dictionary = learn_dictionary(patches_normalized, n_components=n_components)

        # Sparse representation
        sparse_repr = dictionary.transform(patches_normalized)

        # Reconstruct patches from dictionary
        reconstructed = dictionary.inverse_transform(sparse_repr)

        # Compute residuals
        residuals = patches_normalized - reconstructed

        # Train SVR on sparse representation → residuals (one SVR per feature)
        svr_models = []
        predicted_residuals = []
        for i in range(patches.shape[1]):
            y_train = residuals[:, i]
            svr = train_svr(sparse_repr, y_train)
            svr_models.append(svr)
            pred = predict_residuals(svr, sparse_repr)
            predicted_residuals.append(pred)

        predicted_residuals = np.array(predicted_residuals).T  # shape: (n_samples, patch_size^2)

        # Add residuals back
        refined_patches_normalized = reconstructed + predicted_residuals

        # Inverse normalization
        refined_patches = scaler.inverse_transform(refined_patches_normalized)

        # Aggregate patches into full image
        upsampled = aggregate_patches(upsampled.shape, patch_size, stride, refined_patches, positions)

    return upsampled

# 7. Full Super-Resolution Pipeline
def super_resolve_image(image_path, high_res_path=None, scale_factor=2, patch_size=8, stride=4, n_iterations=3, n_components=100):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image '{image_path}' not found.")

    print("Running super-resolution...")
    result = iterative_refinement(image, scale_factor, patch_size, stride, n_iterations, n_components)

    output_path = 'refined_image.jpg'
    cv2.imwrite(output_path, result)
    print(f"Saved refined image as {output_path}")

    if high_res_path:
        target_image = cv2.imread(high_res_path, cv2.IMREAD_GRAYSCALE)
        if target_image is None:
            raise FileNotFoundError(f"High-res image '{high_res_path}' not found.")
        # Crop target image to match the shape (optional: center crop)
        target_image = cv2.resize(target_image, (result.shape[1], result.shape[0]))
        psnr = cv2.PSNR(target_image, result)
        print(f"PSNR: {psnr:.2f} dB")

    return result

# 8. Run it
if __name__ == "__main__":
    # Input image and (optionally) high-res ground truth
    low_res_img = 'images/low_res_color.jpg'
    high_res_img = 'images/high_res.jpg'  # Optional, for PSNR

    # Run the super-resolution pipeline
    super_resolve_image(low_res_img, high_res_path=high_res_img, scale_factor=2, patch_size=8, stride=4, n_iterations=3)

"""

# RELLY SLOW

import cv2
import numpy as np
from sklearn.decomposition import DictionaryLearning
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 1. Upsampling using Bicubic Interpolation
def upsample_image(image, scale_factor):
    height, width = image.shape[:2]
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

# 2. Patch Extraction
def extract_patches(image, patch_size, stride):
    patches = []
    positions = []
    height, width = image.shape[:2]
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch.flatten())
            positions.append((y, x))
    return np.array(patches), positions

# 3. Patch Aggregation
def aggregate_patches(image_shape, patch_size, stride, patches, positions):
    reconstructed = np.zeros(image_shape, dtype=np.float32)
    weight_map = np.zeros(image_shape, dtype=np.float32)

    idx = 0
    for (y, x) in positions:
        patch = patches[idx].reshape((patch_size, patch_size))
        reconstructed[y:y + patch_size, x:x + patch_size] += patch
        weight_map[y:y + patch_size, x:x + patch_size] += 1
        idx += 1

    weight_map[weight_map == 0] = 1
    return (reconstructed / weight_map).astype(np.uint8)

# 4. Sparse Representation using Dictionary Learning
def learn_dictionary(patches, n_components=100):
    dict_learner = DictionaryLearning(n_components=n_components,
                                      fit_algorithm='lars',
                                      transform_algorithm='lasso_lars',
                                      random_state=0)
    dict_learner.fit(patches)
    return dict_learner

# 5. Support Vector Regression (SVR)
def train_svr(X, y):
    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.01)
    svr_model.fit(X, y)
    return svr_model

def predict_residuals(svr_model, sparse_repr):
    return svr_model.predict(sparse_repr)

# 6. Iterative Refinement
def iterative_refinement(image, scale_factor=2, patch_size=8, stride=4, n_iterations=3, n_components=100, max_dict_patches=10000):
    upsampled = upsample_image(image, scale_factor)
    upsampled = upsampled.astype(np.float32)

    for iter_num in range(n_iterations):
        print(f"Iteration {iter_num + 1}/{n_iterations}...")

        # Extract patches
        patches, positions = extract_patches(upsampled, patch_size, stride)
        print(f"  Extracted {len(patches)} patches")

        # Subsample a smaller set for dictionary learning
        n_samples = min(max_dict_patches, patches.shape[0])
        subset_indices = np.random.choice(patches.shape[0], size=n_samples, replace=False)
        patches_subset = patches[subset_indices]

        # Normalize subset for dictionary learning
        scaler = StandardScaler()
        patches_subset_normalized = scaler.fit_transform(patches_subset)

        # Learn dictionary
        print("  Learning dictionary...")
        dictionary = learn_dictionary(patches_subset_normalized, n_components=n_components)

        # Normalize full patch set
        all_patches_normalized = scaler.transform(patches)

        # Sparse representation
        sparse_repr = dictionary.transform(all_patches_normalized)

        # Reconstruct patches from dictionary
        reconstructed = np.dot(sparse_repr, dictionary.components_)

        # Compute residuals
        residuals = all_patches_normalized - reconstructed

        # Train SVR per feature
        svr_models = []
        predicted_residuals = []
        print("  Training SVRs...")
        for i in range(patches.shape[1]):
            y_train = residuals[:, i]
            svr = train_svr(sparse_repr, y_train)
            svr_models.append(svr)
            pred = predict_residuals(svr, sparse_repr)
            predicted_residuals.append(pred)

        predicted_residuals = np.array(predicted_residuals).T

        # Add predicted residuals back
        refined_patches_normalized = reconstructed + predicted_residuals

        # Inverse normalization
        refined_patches = scaler.inverse_transform(refined_patches_normalized)

        # Aggregate patches into full image
        upsampled = aggregate_patches(upsampled.shape, patch_size, stride, refined_patches, positions)

    return upsampled

# 7. Full Super-Resolution Pipeline
def super_resolve_image(image_path, high_res_path=None, scale_factor=2, patch_size=8, stride=4, n_iterations=3, n_components=100):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image '{image_path}' not found.")

    print(f"Original HR image shape: {image.shape}")
    print("Running super-resolution...")
    result = iterative_refinement(image, scale_factor, patch_size, stride, n_iterations, n_components)

    output_path = 'refined_image.jpg'
    cv2.imwrite(output_path, result)
    print(f"Saved refined image as {output_path}")

    if high_res_path:
        target_image = cv2.imread(high_res_path, cv2.IMREAD_GRAYSCALE)
        if target_image is None:
            raise FileNotFoundError(f"High-res image '{high_res_path}' not found.")
        target_image = cv2.resize(target_image, (result.shape[1], result.shape[0]))
        psnr = cv2.PSNR(target_image, result)
        print(f"PSNR: {psnr:.2f} dB")

    return result

# 8. Run it
if __name__ == "__main__":
    low_res_img = 'images/low_res_color.jpg'
    high_res_img = 'images/high_res.jpg'  # Optional

    super_resolve_image(
        low_res_img,
        high_res_path=high_res_img,
        scale_factor=2,
        patch_size=8,
        stride=4,
        n_iterations=3,
        n_components=100
    )




# great. now this is part of my project wher implenet the code. i want you to create a very detailed summary of evething we created and the diffeence from the real thing.

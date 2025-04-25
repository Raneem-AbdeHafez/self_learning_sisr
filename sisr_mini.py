import cv2
import time
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

# 6. Minibatch Learning: Iterative Refinement with Mini-Batches
def iterative_refinement_minibatch(image, scale_factor=2, patch_size=8, stride=4, n_iterations=3, n_components=100, max_dict_patches=10000, batch_size=100):
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

        # Initialize minibatches for processing
        num_batches = len(patches) // batch_size
        residuals = np.zeros_like(all_patches_normalized)
        all_reconstructed = np.zeros_like(all_patches_normalized)


        print("  Processing mini-batches...")
        for batch_idx in range(num_batches + 1):
            # Process each batch
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(patches))

            if batch_start >= batch_end:
                break

            batch_patches = all_patches_normalized[batch_start:batch_end]
            sparse_repr = dictionary.transform(batch_patches)

            # Reconstruct patches from dictionary
            reconstructed = np.dot(sparse_repr, dictionary.components_)
            all_reconstructed[batch_start:batch_end] = reconstructed

            # Compute residuals for the batch
            residual_batch = batch_patches - reconstructed

            # Train SVR per feature for the batch
            svr_models = []
            predicted_residuals = []
            for i in range(batch_patches.shape[1]):
                y_train = residual_batch[:, i]
                svr = train_svr(sparse_repr, y_train)
                svr_models.append(svr)
                pred = predict_residuals(svr, sparse_repr)
                predicted_residuals.append(pred)

            predicted_residuals = np.array(predicted_residuals).T

            # Store the residuals for later aggregation
            residuals[batch_start:batch_end] = predicted_residuals

        # Add predicted residuals back
        refined_patches_normalized = all_reconstructed + residuals

        # Inverse normalization
        refined_patches = scaler.inverse_transform(refined_patches_normalized)

        # Aggregate patches into full image
        upsampled = aggregate_patches(upsampled.shape, patch_size, stride, refined_patches, positions)

    return upsampled

# 7. Full Super-Resolution Pipeline (Color Support)
def super_resolve_image(image_path, high_res_path=None, scale_factor=2, patch_size=8, stride=4, n_iterations=3, n_components=100, batch_size=100):
    # Read color image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image '{image_path}' not found.")

    print(f"Original HR image shape: {image.shape}")
    print("Running super-resolution...")

    # Split channels
    b, g, r = cv2.split(image)

    # Apply super-resolution to each channel
    print("Processing Blue channel...")
    up_b = iterative_refinement_minibatch(b, scale_factor, patch_size, stride, n_iterations, n_components, batch_size=batch_size)

    print("Processing Green channel...")
    up_g = iterative_refinement_minibatch(g, scale_factor, patch_size, stride, n_iterations, n_components, batch_size=batch_size)

    print("Processing Red channel...")
    up_r = iterative_refinement_minibatch(r, scale_factor, patch_size, stride, n_iterations, n_components, batch_size=batch_size)

    # Merge channels
    result = cv2.merge([up_b, up_g, up_r])

    output_path = 'refined_image_color_minibatch.jpg'
    cv2.imwrite(output_path, result)
    print(f"Saved refined image as {output_path}")

    # PSNR calculation (if high-res provided)
    if high_res_path:
        target_image = cv2.imread(high_res_path, cv2.IMREAD_COLOR)
        if target_image is None:
            raise FileNotFoundError(f"High-res image '{high_res_path}' not found.")
        target_image = cv2.resize(target_image, (result.shape[1], result.shape[0]))

        # Compute average PSNR over all channels
        psnr_b = cv2.PSNR(target_image[:, :, 0], result[:, :, 0])
        psnr_g = cv2.PSNR(target_image[:, :, 1], result[:, :, 1])
        psnr_r = cv2.PSNR(target_image[:, :, 2], result[:, :, 2])
        avg_psnr = (psnr_b + psnr_g + psnr_r) / 3

        print(f"Average PSNR: {avg_psnr:.2f} dB")

    return result

if __name__ == "__main__":
    import time
    start_time = time.time()

    low_res_img = 'images/low_res_color.jpg'
    high_res_img = 'images/high_res.jpg'  

    result = super_resolve_image(
        low_res_img,
        high_res_path=high_res_img,
        scale_factor=2,
        patch_size=8,
        stride=4,
        n_iterations=3,
        n_components=100,
        batch_size=100  
    )

    elapsed_time = time.time() - start_time
    print(f"\nTotal time taken: {elapsed_time:.2f} seconds")
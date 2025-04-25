import numpy as np
import time
from sklearn.svm import SVR
from sklearn.decomposition import DictionaryLearning
from skimage.util import view_as_windows
from skimage import img_as_float
import cv2

def extract_patches(image, patch_size=8):
    """
    Extracts patches of size patch_size x patch_size from an image.
    """
    patches = view_as_windows(image, (patch_size, patch_size, image.shape[2]), step=patch_size)
    return patches.reshape(-1, patch_size * patch_size * image.shape[2])

def learn_dictionary(patches, n_components=100):
    """
    Learns a dictionary from the given image patches using Dictionary Learning.
    """
    dict_learner = DictionaryLearning(n_components=n_components)
    dictionary = dict_learner.fit(patches)
    return dictionary

def train_svr_model(low_res_patches, high_res_patches):
    """
    Train an SVR model to map low-resolution patches to high-resolution patches.
    """
    # Print the shapes of the patches
    print(f"Low res patches shape: {low_res_patches.shape}")
    print(f"High res patches shape: {high_res_patches.shape}")
    
    # Ensure both low_res_patches and high_res_patches have the same number of patches
    if low_res_patches.shape[0] != high_res_patches.shape[0]:
        raise ValueError(f"Number of low-res patches ({low_res_patches.shape[0]}) does not match the number of high-res patches ({high_res_patches.shape[0]})")

    # Flatten high_res_patches from 2D array (patches) to 1D per patch
    num_samples, patch_size = high_res_patches.shape
    high_res_patches_flat = high_res_patches.reshape(num_samples, -1)  # Flatten each patch

    # Now, we flatten the high-resolution patches further to make it 1D for each patch's target value
    high_res_patches_flat = high_res_patches_flat.reshape(-1)

    # Train the SVR model
    svr_model = SVR(kernel='linear')
    svr_model.fit(low_res_patches, high_res_patches_flat)
    return svr_model



def reconstruct_image(patches, image_shape, patch_size=8, step=8):
    """
    Reconstruct the image from patches.
    """
    rows, cols, channels = image_shape
    img_reconstructed = np.zeros((rows, cols, channels))
    patch_idx = 0

    for r in range(0, rows - patch_size + 1, step):
        for c in range(0, cols - patch_size + 1, step):
            img_reconstructed[r:r+patch_size, c:c+patch_size, :] = patches[patch_idx].reshape(patch_size, patch_size, channels)
            patch_idx += 1

    return img_reconstructed

def refine_image(low_res_image, svr_model, dictionary, patch_size=8, n_iter=10):
    """
    Iteratively refine the high-resolution image.
    """
    refined_image = np.copy(low_res_image)

    for _ in range(n_iter):
        refined_image_channels = []

        for channel in range(refined_image.shape[2]):  # Process each color channel separately
            low_res_channel = refined_image[:, :, channel]
            low_res_patches = extract_patches(low_res_channel, patch_size)
            
            # Use the dictionary to represent the patches
            low_res_patches_dict = dictionary.transform(low_res_patches)
            
            # Predict high-resolution patches using the trained SVR model
            high_res_patches = svr_model.predict(low_res_patches_dict)
            
            # Reconstruct the high-resolution image from the predicted patches
            high_res_channel = reconstruct_image(high_res_patches, refined_image.shape, patch_size)
            refined_image_channels.append(high_res_channel)

        # Combine all color channels back into one image
        refined_image = np.stack(refined_image_channels, axis=-1)

    return refined_image

def main():
    # Start timer for the entire process
    start_time = time.time()

    # Load low-resolution and high-resolution images
    low_res_image = cv2.imread('images/low_res_color.jpg')
    high_res_image = cv2.imread('images/high_res.jpg')

    # Check if the images were loaded properly
    if low_res_image is None:
        print("Error: Low-resolution image not found or unable to load.")
        return
    if high_res_image is None:
        print("Error: High-resolution image not found or unable to load.")
        return

    # Convert images to float for better precision in calculations
    low_res_image = img_as_float(low_res_image)
    high_res_image = img_as_float(high_res_image)

    # Measure time for patch extraction
    patch_start_time = time.time()
    low_res_patches = extract_patches(low_res_image)
    high_res_patches = extract_patches(high_res_image)
    print(f"Patch extraction took {time.time() - patch_start_time:.4f} seconds.")

    # Learn a dictionary from the low-resolution patches
    dict_start_time = time.time()
    dictionary = learn_dictionary(low_res_patches)
    print(f"Dictionary learning took {time.time() - dict_start_time:.4f} seconds.")

    # Train the SVR model using the learned dictionary
    svr_start_time = time.time()
    svr_model = train_svr_model(low_res_patches, high_res_patches)
    print(f"SVR training took {time.time() - svr_start_time:.4f} seconds.")

    # Refine the high-resolution image
    refine_start_time = time.time()
    high_res_result = refine_image(low_res_image, svr_model, dictionary)
    print(f"Refinement took {time.time() - refine_start_time:.4f} seconds.")

    # Save the resulting super-resolved image
    cv2.imwrite('super_resolved_image.jpg', np.uint8(high_res_result * 255))

    print("Super-resolution completed. The result is saved as 'super_resolved_image.jpg'.")
    
    # Total time taken for the whole process
    print(f"Total time for super-resolution: {time.time() - start_time:.4f} seconds.")
 

if __name__ == "__main__":
    main()

import numpy as np
import cv2
import os
from aicspylibczi import CziFile
from skimage.io import imsave

# Define the directories
czi_dir = 'czi_files/'
adipose_mask_dir = 'training_data/adipose_masks/'
nuclei_mask_dir = 'training_data/nuclei_masks/'
results_dir = 'results/'

# Ensure the mask directories exist
os.makedirs(adipose_mask_dir, exist_ok=True)
os.makedirs(nuclei_mask_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Manual threshold value
manual_threshold_value = 50

# Helper function to ensure the image is 2D
def ensure_2d_image(image):
    if image.ndim == 5:
        return image[0, 0, 0, :, :]
    elif image.ndim == 3:
        return image[:, :, 0]  # Assuming it's a single-channel 3D image
    elif image.ndim == 2:
        return image
    else:
        raise ValueError(f"Unexpected image dimensions: {image.shape}")

# Process each CZI file
for file_name in os.listdir(czi_dir):
    if file_name.endswith('.czi'):
        # Full path to the CZI file
        czi_path = os.path.join(czi_dir, file_name)

        # Load the CZI file using aicspylibczi
        czi = CziFile(czi_path)

        # Extract image data - assuming you want the first scene, first channel, and middle Z-plane
        image_data = czi.read_mosaic(C=0, M=0, Z=0, T=0)

        # Convert image data to 2D grayscale
        gray = ensure_2d_image(image_data)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(gray)

        # Apply manual thresholding
        _, adipose_thresh = cv2.threshold(enhanced_image, manual_threshold_value, 255, cv2.THRESH_BINARY)

        # Morphological Operations: Opening (remove small objects) and Closing (fill gaps)
        kernel = np.ones((3, 3), np.uint8)
        adipose_mask = cv2.morphologyEx(adipose_thresh, cv2.MORPH_OPEN, kernel)
        adipose_mask = cv2.morphologyEx(adipose_mask, cv2.MORPH_CLOSE, kernel)

        # Save the post-processed adipose mask
        adipose_mask_path = os.path.join(adipose_mask_dir, f'{os.path.splitext(file_name)[0]}_adipose.npy')
        np.save(adipose_mask_path, adipose_mask)

        # Save the adipose mask image as TIFF
        adipose_image_path = os.path.join(results_dir, f'{os.path.splitext(file_name)[0]}_adipose.tiff')
        imsave(adipose_image_path, adipose_mask)

        # Repeat similar steps for the nuclei mask
        _, nuclei_thresh = cv2.threshold(enhanced_image, manual_threshold_value, 255, cv2.THRESH_BINARY_INV)
        nuclei_mask = cv2.morphologyEx(nuclei_thresh, cv2.MORPH_OPEN, kernel)
        nuclei_mask = cv2.morphologyEx(nuclei_mask, cv2.MORPH_CLOSE, kernel)

        # Save the post-processed nuclei mask
        nuclei_mask_path = os.path.join(nuclei_mask_dir, f'{os.path.splitext(file_name)[0]}_nuclei.npy')
        np.save(nuclei_mask_path, nuclei_mask)

        # Save the nuclei mask image as TIFF
        nuclei_image_path = os.path.join(results_dir, f'{os.path.splitext(file_name)[0]}_nuclei.tiff')
        imsave(nuclei_image_path, nuclei_mask)

        print(f'Processed {file_name}')

print('Processing complete!')

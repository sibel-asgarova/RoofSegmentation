import os
import numpy as np
from PIL import Image
import gc

def save_matrix_if_not_exists(matrix, save_path):
    """ Save matrix to a file if it doesn't already exist. """
    if not os.path.exists(save_path):  # Check if file already exists
        np.save(save_path, matrix)  # Save the matrix if it doesn't exist
        print(f"Matrix saved to {save_path}")
    else:
        print(f"File {save_path} already exists. Skipping save.")


def create_and_save_all_image_matrices(image_list, save_dir):
    """
    Create and save all image matrices as a single numpy file. Images are resized to 512x512.

    Args:
        image_list (list): List of image file paths.
        save_dir (str): Directory to save the numpy array.
    """
    all_image_matrices = []
    
    # Iterate over all images in the list
    for image_file in image_list:
        try:
            image_path = image_file
            # Load the image
            image = Image.open(image_path)

            # Resize the image to 512x512
            image = image.resize((512, 512))

            # Convert the image to a numpy array
            image_matrix = np.array(image)

            # Append the image matrix to the list
            all_image_matrices.append(image_matrix)
            
        except Exception as e:
            print(f"Error loading or processing image {image_file}: {e}")
        
        # Optional: Trigger garbage collection to free memory between processing
        gc.collect()

    # Check if there are valid image matrices
    if all_image_matrices:
        # Convert the list of matrices into a single numpy array
        all_images_array = np.array(all_image_matrices)

        # Save the combined array as a single .npy file using the save_matrix_if_not_exists function
        save_path = os.path.join(save_dir, "all_images_matrices.npy")
        save_matrix_if_not_exists(all_images_array, save_path)  # Save if not exists
    else:
        print("No valid images were processed.")


def create_and_save_all_mask_matrices(mask_list, save_dir):
    """
    Create and save all mask matrices as a single numpy file. Masks are resized to 512x512.

    Args:
        mask_list (list): List of mask file paths.
        save_dir (str): Directory to save the numpy array.
    """
    all_mask_matrices = []
    
    # Iterate over all masks in the list
    for mask_file in mask_list:
        try:
            mask_path = mask_file
            # Load the mask
            mask = Image.open(mask_path)

            # Resize the mask to 512x512
            mask = mask.resize((512, 512))

            # Convert the mask to a numpy array
            mask_matrix = np.array(mask)

            # Append the mask matrix to the list
            all_mask_matrices.append(mask_matrix)
        
        except Exception as e:
            print(f"Error loading or processing mask {mask_file}: {e}")
        
        # Optional: Trigger garbage collection to free memory between processing
        gc.collect()

    # Check if there are valid mask matrices
    if all_mask_matrices:
        # Convert the list of matrices into a single numpy array
        all_masks_array = np.array(all_mask_matrices)

        # Save the combined array as a single .npy file using the save_matrix_if_not_exists function
        save_path = os.path.join(save_dir, "all_masks_matrices.npy")
        save_matrix_if_not_exists(all_masks_array, save_path)  # Save if not exists
    else:
        print("No valid masks were processed.")

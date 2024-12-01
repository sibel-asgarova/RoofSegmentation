import os

# Function to get image paths
def get_image_paths(path_to_images):
       """
    Retrieves the filenames of all files in the specified directory.

    Parameters:
    path_to_images (str): The directory path where images are stored.

    Returns:
    list: A list containing the names of all files in the directory.
    """

    return [name for name in os.listdir(path_to_images)]

# Function to generate mask paths based on image paths

def get_mask_paths(directory_images):
     """
    Generates mask filenames based on image filenames by replacing the file extension 
    with '_vis.tif'. Assumes that the mask file names follow this structure.

    Parameters:
    directory_images (list): A list of image filenames.

    Returns:
    list: A list of gene
    rated mask filenames corresponding to each image.
    """
    directory_masks = []
    for path in directory_images:
        path_mask = str(path.split('.')[0]) + '_vis.tif'
        directory_masks.append(path_mask)
    return directory_masks



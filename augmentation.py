import numpy as np
import albumentations as A

def augment_image_and_mask(image, mask):
    """
    Apply data augmentation to both the image and the mask.
    
    Args:
        image (numpy.ndarray): Image to augment.
        mask (numpy.ndarray): Mask to augment.
        
    Returns:
        augmented_image (numpy.ndarray): Augmented image.
        augmented_mask (numpy.ndarray): Augmented mask.
    """
    # Define augmentations
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomSizedCrop(min_max_height=(300, 400), height=512, width=512, p=0.5),
        A.GaussianBlur(p=0.2),
    ])

    # Apply transformations
    augmented = transform(image=image, mask=mask)
    augmented_image = augmented['image']
    augmented_mask = augmented['mask']

    return augmented_image, augmented_mask

def apply_augmentation_to_arrays(image_arrays, mask_arrays, augmentations_per_pair=1):
    """
    Apply augmentation to all image and mask pairs in the arrays.
    
    Args:
        image_arrays (numpy.ndarray): Array of images to augment.
        mask_arrays (numpy.ndarray): Array of masks to augment.
        augmentations_per_pair (int): Number of augmentations to apply per image-mask pair.
        
    Returns:
        augmented_images (numpy.ndarray): Augmented images.
        augmented_masks (numpy.ndarray): Augmented masks.
    """
    augmented_images = []
    augmented_masks = []

    # Iterate over the arrays and apply augmentation
    for i in range(len(image_arrays)):
        image = image_arrays[i]
        mask = mask_arrays[i]

        for _ in range(augmentations_per_pair):
            augmented_image, augmented_mask = augment_image_and_mask(image, mask)
            augmented_images.append(augmented_image)
            augmented_masks.append(augmented_mask)

    # Convert the list of augmented images and masks to numpy arrays
    augmented_images = np.array(augmented_images)
    augmented_masks = np.array(augmented_masks)

    return augmented_images, augmented_masks

# Example of how to load the .npy files and apply augmentation
def load_data(image_path, mask_path):
    """
    Load image and mask matrices from .npy files.
    
    Args:
        image_path (str): Path to the saved image matrices (.npy).
        mask_path (str): Path to the saved mask matrices (.npy).
        
    Returns:
        images (numpy.ndarray): Loaded image matrix array.
        masks (numpy.ndarray): Loaded mask matrix array.
    """
    images = np.load(image_path)  # Load images matrix from .npy file
    masks = np.load(mask_path)    # Load masks matrix from .npy file
    return images, masks




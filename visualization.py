import os
import matplotlib.pyplot as plt
from PIL import Image

def visualize_single_image_and_mask(image_path, mask_path):
    """
    Visualize a single image and its corresponding mask side by side.

    Args:
        image_path (str): Path to the image file.
        mask_path (str): Path to the mask file.
    """
    # Load the image and mask
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the image
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    # Display the mask
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

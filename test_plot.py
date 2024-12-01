import matplotlib.pyplot as plt

def plot_image_and_masks(model, X_test, y_test, index=0):
    """
    Function to display an image, its true mask, and the predicted mask.

    Parameters:
    - model: Trained model used to predict the mask.
    - X_test: Test images dataset.
    - y_test: True masks for the test dataset.
    - index: Index of the sample to display.
    """
    # Get the image, true mask, and predicted mask
    image = X_test[index]
    true_mask = y_test[index]
    predicted_mask = model.predict(X_test[index:index+1])[0]
    
    # Plotting the image, true mask, and predicted mask
    plt.figure(figsize=(12, 4))

    # Plot the image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis('off')

    # Plot the true mask (as black and white)
    plt.subplot(1, 3, 2)
    plt.imshow(true_mask.squeeze(), cmap='gray')  # Squeeze to remove extra dimension
    plt.title("True Mask")
    plt.axis('off')

    # Plot the predicted mask (as black and white)
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask.squeeze(), cmap='gray')  # Squeeze to remove extra dimension
    plt.title("Predicted Mask")
    plt.axis('off')

    # Show the plot
    plt.show()

# Example usage:
# plot_image_and_masks(model, X_test, y_test, index=0)

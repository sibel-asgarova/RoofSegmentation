from sklearn.model_selection import train_test_split

def split_data(all_images, all_masks, train_size=0.7, val_size=0.2, test_size=0.1):
    """
    Split the augmented image and mask data into training, validation, and testing sets.
    
    Args:
        all_images (numpy.ndarray): All image data (original + augmented).
        all_masks (numpy.ndarray): All mask data (original + augmented).
        train_size (float): Fraction of data to use for training (default is 70%).
        val_size (float): Fraction of data to use for validation (default is 20%).
        test_size (float): Fraction of data to use for testing (default is 10%).
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test: Training, validation, and testing sets.
    """
    # Split into train and temp (for validation and testing)
    X_train, X_temp, y_train, y_temp = train_test_split(all_images, all_masks, test_size=1 - train_size, random_state=42)
    
    # Calculate adjusted validation size based on the remaining data (val_size + test_size should add up to 1 - train_size)
    adjusted_val_size = val_size / (val_size + test_size)
    
    # Split the temp set into validation and test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 - adjusted_val_size, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


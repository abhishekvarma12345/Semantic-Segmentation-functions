import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import torch

def mask_encoding(RGB_mask, color_maps, one_hot_encoded=False):
    """
    Encodes a mask into a label mask or a one-hot encoded mask based on a color map.

    Parameters:
    RGB_mask (np.ndarray): The mask to be encoded. It should be a 3D NumPy array where each element is an RGB tuple.
    color_maps (dict): A dictionary mapping RGB tuples to labels. Each key is a tuple representing an RGB color, 
                       and each value is a tuple where the first element is the label for that color.
    one_hot_encoded (bool): If True, the function returns a one-hot encoded mask. If False, the function returns a label mask. 
                            Default is False.

    Returns:
    output_mask (np.ndarray): If one_hot_encoded is False, returns a 2D NumPy array where each element is the label of the corresponding pixel in the input image. 
                              If one_hot_encoded is True, returns an array of 2D boolean arrays representing a one-hot encoded mask.
    """
    output_mask = []
    if one_hot_encoded:
        ohe_mask = []
        for rgb, (label, _) in color_maps.items():
            cmap = np.all(np.equal(RGB_mask, rgb), axis=-1)
            ohe_mask.append(cmap)
        output_mask = np.array(ohe_mask, dtype=np.int32)
    else:
        label_mask = np.zeros(RGB_mask.shape[:2], dtype=np.int32)
        for rgb, (label, _) in color_maps.items():
            match_mask = (RGB_mask == np.array(rgb)).all(axis=-1)
            label_mask[match_mask] = label
        label_mask = np.expand_dims(label_mask, axis=2)
        output_mask = label_mask

    return output_mask

def plot_random_masks(mask_paths, color_maps, num_masks=5):
    """
    Plots original masks and their corresponding label masks for a random selection of mask paths.

    Parameters:
    mask_paths (list): A list of paths to the mask images.
    color_maps (dict): A dictionary mapping RGB tuples to labels. Each key is a tuple representing an RGB color, 
                       and each value is a tuple where the first element is the label for that color.
    num_masks (int, optional): The number of masks to plot. Default is 5.

    Returns:
    None. This function shows the plots using matplotlib's plt.show() function.
    """
    # Select a random subset of the mask paths
    selected_paths = random.sample(mask_paths, num_masks)
    # Create a figure for the plots
    fig, axs = plt.subplots(num_masks, 2, figsize=(10, num_masks*5))
    
    for i, mask_path in enumerate(selected_paths):
        # Convert the mask to labels
        label_mask = mask_encoding(mask_path, color_maps)
        
        # Load the original mask
        BGR_mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        RGB_mask = cv2.cvtColor(BGR_mask, cv2.COLOR_BGR2RGB)
        
        # Plot the original mask and the label mask
        axs[i, 0].imshow(RGB_mask)
        axs[i, 0].set_title(f'Original Mask \n shape: {RGB_mask.shape}')
        axs[i, 1].imshow(label_mask)
        axs[i, 1].set_title(f'Label Mask \n shape: {label_mask.shape}')
    
    # Show the plots
    plt.tight_layout()
    plt.show()

def find_unique_values(mask_paths):
    """
    Finds and returns the unique RGB color values present in a list of image masks.

    Parameters:
    mask_paths (list): A list of file paths to the image masks.

    Returns:
    list: A sorted list of unique RGB color values present in the image masks. Each color value is represented as a tuple of three integers.

    This function reads each image mask using OpenCV, converts it from BGR to RGB color space, and then finds the unique color values in the mask. 
    The unique color values from all masks are collected in a set to remove duplicates. Finally, the function returns a sorted list of unique color values.
    """
    unique_values = set()

    for mask_path in mask_paths:
        # Load the image using OpenCV
        BGR_segmentation_mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        RGB_segmentation_mask = cv2.cvtColor(BGR_segmentation_mask, cv2.COLOR_BGR2RGB)

        # Collect unique color maps from the segmentation mask
        unique_values.update(set(map(tuple,RGB_segmentation_mask.reshape(-1,3))))

    return sorted(list(unique_values))

def plot_images_masks(images, masks, one_hot_encoded=None):
    """
    Plots a batch of images and their corresponding masks.

    Parameters:
    images (torch.Tensor): A 4D tensor containing a batch of images. The dimensions should be (batch_size, height, width, channels).
    masks (torch.Tensor): A 4D tensor containing a batch of masks. The dimensions should be (batch_size, num_classes, height, width) if one_hot_encoded is True, 
                          otherwise the dimensions should be (batch_size, height, width).
    one_hot_encoded (bool, optional): If True, the function assumes that the masks are one-hot encoded and takes the argmax over the class dimension for plotting. 
                                      If False or None, the function directly plots the masks. Default is None.

    This function assumes that the input images and masks are PyTorch tensors. It converts these tensors to numpy arrays for plotting. 
    The function creates a 2x8 grid of subplots, where the top row displays the images and the bottom row displays the corresponding masks. 
    The function uses matplotlib for plotting.
    """
    # Assuming images and masks are PyTorch tensors
    # Convert tensors to numpy arrays for plotting
    images = images.permute(0, 2, 3, 1).numpy()

    fig, axs = plt.subplots(2, 8, figsize=(20, 5))

    for i in range(8):  # Assuming batch size is 8
        axs[0, i].imshow(images[i])
        axs[0, i].axis('off')
        axs[0, i].set_title(f'Image {i+1}')

        # Create a color map for the mask
        if one_hot_encoded:
            mask = torch.argmax(masks[i], dim=0).numpy()
        else:
            mask = masks[i].permute(1, 2, 0).numpy() # 2D array of labels with an extra dimension
        axs[1, i].imshow(mask)
        axs[1, i].axis('off')
        axs[1, i].set_title(f'Mask {i+1}')

    plt.tight_layout()
    plt.show()


def plot_masks(model, dataloader, device=None):
    """
    Plots the true and predicted masks for a batch of images from a data loader.

    Parameters:
    model (torch.nn.Module): The PyTorch model to use for prediction.
    dataloader (torch.utils.data.DataLoader): The data loader providing the batches of images and true masks.
    device (torch.device, optional): The device on which the model and data are. If not provided, it will use the device of the model.

    This function sets the model to evaluation mode and disables gradient tracking. It then retrieves a batch of images and true masks from the data loader and uses the model to predict the masks for the images. 
    The function plots the true and predicted masks for each image in the batch. 
    The true and predicted masks are displayed side by side for easy comparison. 
    The function assumes that the model and data loader are compatible and that the model is on the same device as the images.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients
        for i, (images, true_masks) in enumerate(dataloader):
            if i == 1:  # Only one batch of images is needed
                break
            pred_masks = model(images.to(device))  # Forward pass to get the predicted masks
            for j in range(images.size(0)):  # For each image in the batch
                fig, ax = plt.subplots(1, 2)  # Create a new figure with two subplots
                ax[0].imshow(true_masks[j].argmax(dim=0).unsqueeze(dim=2).numpy())  # Plot the true mask
                ax[0].title.set_text('True Mask')
                ax[1].imshow(pred_masks[j].argmax(dim=0).unsqueeze(dim=2).cpu())  # Plot the predicted mask
                ax[1].title.set_text('Predicted Mask')
                plt.show()



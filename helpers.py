def rgb_to_ohe_mask(rgb_mask, colorcodes):
    """
    Converts an RGB mask to a one-hot encoded (OHE) mask.

    Parameters:
    rgb_mask (numpy.ndarray): An input mask in RGB format. It is a 3D array where the third dimension represents the RGB channels.
    colorcodes (list): A list of color codes. Each color code is a list of three integers representing an RGB color.

    Returns:
    numpy.ndarray: A 3D array representing the OHE mask. The third dimension has length equal to the number of color codes. Each slice along the third dimension is a binary mask corresponding to one color code.
    """
    output_mask = []

    for color in colorcodes:
        cmap = np.all(np.equal(rgb_mask, color), axis=-1)
        output_mask.append(cmap)

    output_mask = np.stack(output_mask, axis=-1)
    return output_mask



def find_unique_values(mask_paths):
    """
    Finds and returns the unique RGB color values present in a list of image masks.

    Parameters:
    mask_paths (list): A list of file paths to the image masks.

    Returns:
    list: A sorted list of unique RGB color values present in the image masks. Each color value is represented as a tuple of three integers.

    This function reads each image mask using OpenCV, converts it from BGR to RGB color space, and then finds the unique color values in the mask. The unique color values from all masks are collected in a set to remove duplicates. Finally, the function returns a sorted list of unique color values.
    """
    unique_values = set()

    for mask_path in mask_paths:
        # Load the image using OpenCV
        BGR_segmentation_mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        RGB_segmentation_mask = cv2.cvtColor(BGR_segmentation_mask, cv2.COLOR_BGR2RGB)

        # Collect unique color maps from the segmentation mask
        unique_values.update(set(map(tuple,RGB_segmentation_mask.reshape(-1,3))))

    return sorted(list(unique_values))

def plot_images_masks(images, masks):
    """
    Plots a batch of images and their corresponding masks.

    Parameters:
    images (torch.Tensor): A 4D tensor containing a batch of images. The dimensions should be (batch_size, height, width, channels).
    masks (torch.Tensor): A 4D tensor containing a batch of masks. The dimensions should be (batch_size, num_classes, height, width).

    This function assumes that the input images and masks are PyTorch tensors. It converts these tensors to numpy arrays for plotting. The function creates a 2x8 grid of subplots, where the top row displays the images and the bottom row displays the corresponding masks. Each mask is created by taking the argmax over the class dimension of the mask tensor. The function uses matplotlib for plotting.
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
        mask = torch.argmax(masks[i], dim=0).numpy()
        axs[1, i].imshow(mask)
        axs[1, i].axis('off')
        axs[1, i].set_title(f'Mask {i+1}')

    plt.tight_layout()
    plt.show()

def plot_masks(model, dataloader):
    """
    Plots the true and predicted masks for a batch of images from a data loader.

    Parameters:
    model (torch.nn.Module): The PyTorch model to use for prediction.
    dataloader (torch.utils.data.DataLoader): The data loader providing the batches of images and true masks.

    This function sets the model to evaluation mode and disables gradient tracking. It then retrieves a batch of images and true masks from the data loader and uses the model to predict the masks for the images. The function plots the true and predicted masks for each image in the batch. The true and predicted masks are displayed side by side for easy comparison. The function assumes that the model and data loader are compatible and that the model is on the same device as the images.
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


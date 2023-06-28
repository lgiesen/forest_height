import matplotlib.pyplot as plt
import numpy as np


def normalize_color(img):
    assert len(img.shape) == 3
    # Extract Red, Green, and Blue bands
    red = img[2, :, :]
    green = img[1, :, :]
    blue = img[0, :, :]

    # Normalize the bands to [0, 1] range
    red_norm = (red - red.min()) / (red.max() - red.min())
    green_norm = (green - green.min()) / (green.max() - green.min())
    blue_norm = (blue - blue.min()) / (blue.max() - blue.min())

    return np.stack((red_norm, green_norm, blue_norm), axis=-1)

def plot(img):
    if(len(img.shape) > 3):
        print("The image dimensions are larger than 3 and cannot be plotted.")
    #satellite image
    if(len(img.shape) == 3):
        img = normalize_color(img)

    # Plot the image
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap='viridis')
    plt.axis('off')
    plt.show()
     
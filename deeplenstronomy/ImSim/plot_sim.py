import lenstronomy.Plots.plot_util as plot_util
import numpy as np

def plot_single_band(ax, image, scale_max=100):
    """
    Parameters:
        ax (instance): Axes instance from matplotlib
        image (ndarray): Array of image pixel values
        scale_max (float): Sets the color scale of the image

    Returns:
        ax (instance): Axes instance containing the image to be plotted.
    """
    img = np.zeros((image.shape[0], image.shape[1]), dtype=float)
    img[:,:] = plot_util.sqrt(image, scale_min=0, scale_max=100)
    ax.imshow(img, aspect='equal', origin='lower')
    return ax


def plot_three_bands(ax, band_1, band_2, band_3, scale_max=100):
    """
    Parameters:
        ax (instance): Axes instance from matplotlib
        band_1 (ndarray): Array of image pixel values for the first band
        band_2 (ndarray): Array of image pixel values for the second band
        band_3 (ndarray): Array of image pixel values for the third band
        scale_max (float): Sets the color scale of the image

    Returns:
        ax (instance): Axes instance containing the image to be plotted.
    """
    img = np.zeros((band_1.shape[0], band_1.shape[1], 3), dtype=float)
    img[:,:,0] = plot_util.sqrt(band_1, scale_min=0, scale_max=100)
    img[:,:,1] = plot_util.sqrt(band_2, scale_min=0, scale_max=100)
    img[:,:,2] = plot_util.sqrt(band_3, scale_min=0, scale_max=100)

    ax.imshow(img, aspect='equal', origin='lower')
    return ax

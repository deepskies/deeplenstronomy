import lenstronomy.Plots.plot_util as plot_util
import numpy as np

def plot_single_band(ax, image, scale_max=100):
    img = np.zeros((image.shape[0], image.shape[1]), dtype=float)
    img[:,:] = plot_util.sqrt(image, scale_min=0, scale_max=100)
    ax.imshow(img, aspect='equal', origin='lower')
    return ax


def plot_three_bands(ax, band_1, band_2, band_3, scale_max=100):
    img = np.zeros((band_1.shape[0], band_1.shape[1], 3), dtype=float)
    img[:,:,0] = plot_util.sqrt(band_1, scale_min=0, scale_max=100)
    img[:,:,1] = plot_util.sqrt(band_2, scale_min=0, scale_max=100)
    img[:,:,2] = plot_util.sqrt(band_3, scale_min=0, scale_max=100)

    ax.imshow(img, aspect='equal', origin='lower')
    return ax

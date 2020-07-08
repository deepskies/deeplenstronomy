import numpy as np
import os
import yaml
from matplotlib import pyplot as plt
import lenstronomy.Plots.plot_util as plot_util



# example images
#   multiband rgb
#   multiple channels
#   plot as a function of parameter values

# plot distributions of all output parameters
# 2-d disributions of correlated parameters
# select quartiles in a parameter


# functions: rescale image


def create_plot_grid():
    """
    Create 2xN plot grid
    """


def histogram2d(ax):
    """
    Create 2xN plot grid
    """


def histogram1d(ax):
    """
    Create 2xN plot grid
    """


def diagnostics():

    # read image data

    # read catalog data

    # create three-band + rgb image of top object in each parameer
    # create 2d histograms for all pairs of parameters
    # create 1d histograms for all indvidual parameters


    for sample in sample_list:




def select_quantile(arr, quantile):
    val = np.quantile(arr



def plot_single_band(ax, image, scale_max=100, title=""):
    """
    Parameters:
        ax (instance): Axes instance from matplotlib
        image (ndarray): Array of image pixel values
        scale_max (float): Sets the color scale of the image
    Returns:
        ax (instance): Axes instance containing the image to be plotted.
    """


    # set up image
    img = np.zeros((image.shape[0], image.shape[1]), dtype=float)

    # scale image
    img[:,:] = plot_util.sqrt(image, scale_min=0, scale_max=scale_max)

    # plot image
    ax.imshow(img, aspect='equal', origin='lower')

    # set title
    ax[i][0].set_title(title)


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

    # set up image
    img = np.zeros((band_1.shape[0], band_1.shape[1], 3), dtype=float)

    # scale all images
    img[:,:,0] = plot_util.sqrt(band_1, scale_min=0, scale_max=scale_max)
    img[:,:,1] = plot_util.sqrt(band_2, scale_min=0, scale_max=scale_max)
    img[:,:,2] = plot_util.sqrt(band_3, scale_min=0, scale_max=scale_max)

    # plot images
    ax.imshow(img, aspect='equal', origin='lower')


    ax[i][2].set_title("Real lens + background")
    bins = np.linspace(-20,20,101)
    ax[i][3].hist(np.ravel(image),histtype='step', lw=4, bins=bins, label = 'fake lens + bg')
    ax[i][3].hist(np.ravel(arc_image),histtype='step', lw=4, bins=bins, label = 'fake lens, real bg')
    ax[i][3].hist(np.ravel(des_candidate),histtype='step', lw=4, bins=bins, label = 'real lens + bg')
    ax[i][3].set_title("pixel histogram")
    ax[i][3].legend()

    return ax





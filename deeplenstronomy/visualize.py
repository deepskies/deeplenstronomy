"""Functions to visualize images."""

from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def _no_stretch(val):
    return val

def view_image(image, stretch_func=_no_stretch, **imshow_kwargs):
    """
    Plot an image.

    Args:
        image (array):  a 2-dimensional array of pixel values OR a list-like object of 2-dimensional arrays of pixel values 
        stretch_func (func, optional, default=pass): stretching function to apply to pixel values (e.g. np.log10)
        imshow_kwargs (dict): dictionary of keyword arguments and their values to pass to matplotlib.pyplot.imshow 
    """
    if len(np.shape(image)) > 2:
        #multi-band mode
        fig, axs = plt.subplots(1, np.shape(image)[0])
        for index, single_band_image in enumerate(image):
            axs[index].imshow(stretch_func(single_band_image), **imshow_kwargs)
            axs[index].set_xticks([], [])
            axs[index].set_yticks([], [])

        fig.tight_layout()
        plt.show(block=True)
        plt.close()

    else:
        #simgle-band mode
        plt.figure()
        plt.imshow(stretch_func(image), **imshow_kwargs)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.show(block=True)
        plt.close()

    return

def view_image_rgb(images, Q=2.0, stretch=4.0, **imshow_kwargs):
    """
    Merge images into a single RGB image. This function assumes the image array
    is ordered [g, r, i].

    Args:
        images (List[np.array]): a list of at least 3 2-dimensional arrays of pixel values corresponding to different photometric bandpasses
        imshow_kwargs (dict): dictionary of keyword arguments and their values to pass to matplotlib.pyplot.imshow
    """

    assert len(images) > 2, "3 images are needed to generate an RGB image"
    
    rgb = make_lupton_rgb(images[2],
                          images[1],
                          images[0],
                          Q=Q, stretch=stretch)

    plt.figure()
    plt.imshow(rgb, **imshow_kwargs)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.show(block=True)
    plt.close()

    return

def view_corner(metadata, labels, hist_kwargs={}, hist2d_kwargs={}, label_kwargs={}):
    """
    Show a corner plot of the columns in a DataFrame.
    
    Args:
        metadata (pd.DataFrame): A pandas DataFrame containing the metadata to visualize
        labels (dict): A dictionary mapping column names to axis labels 
        hist_kwargs (dict): keyword arguments to pass to matplotlib.pyplot.hist
        hist2d_kwargs (dict): keyword arguments to pass to matplotlib.pyplot.hist2d
        label_kwargs (dict): keyword arguments to pass to matplotlib.axes.Axes.set_xlabel (and ylabel)
        
    Raises:
        KeyError: if one or more of the columns are not present in the metadata
        TypeError: if metadata is not a pandas DataFrame
        TypeError: if labels is not a dict
    """
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("first argument must be a pandas DataFrame")
        
    if not isinstance(labels, dict):
        raise TypeError("second argument must be a list")
    
    if any([x not in metadata.columns for x in labels]):
        raise KeyError("One or more passed columns is not present in the metadata")
    
    fig, axs = plt.subplots(len(labels), len(labels), figsize=(14,14))

    for row, row_label in enumerate(labels.keys()):
        for col, col_label in enumerate(labels.keys()):

            if row == col:
                # hist
                axs[row, col].hist(metadata[row_label].values, **hist_kwargs)

            elif row > col:
                # hist2d
                axs[row, col].hist2d(metadata[col_label].values, 
                                     metadata[row_label].values, **hist2d_kwargs)
            else:
                axs[row, col].set_visible(False)

            if row == len(labels) -1:
                axs[row, col].set_xlabel(labels[col_label], **label_kwargs)

            if col == 0 and row != 0:
                axs[row, col].set_ylabel(labels[row_label], **label_kwargs)

    fig.tight_layout()
    plt.show()
    plt.close()

# A collection of image visualization functions

from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
import numpy as np

def no_stretch(val):
    return val

def view_image(image, stretch_func=no_stretch, **imshow_kwargs):
    """
    Plot an image.

    :param image: a 2-dimensional array of pixel values
                  OR
                  a list-like object of 2-dimensional arrays of pixel values
    :param stretch_func: func, stretching function to apply to pixel values (e.g. np.log10)
    :param imshow_kwargs: keyword arguments to pass to matplotlib.pyplot.imshow
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
    is ordered [g, r, i]

    :param images: a list of at least 3 2-dimensional arrays of pixel values
                   corresponding to different photometric bandpasses
    :param imshow_kwargs: keyword arguments to pass to matplotlib.pyplot.imshow
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

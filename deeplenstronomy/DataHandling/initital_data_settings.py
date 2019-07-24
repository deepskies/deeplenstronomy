#initial data settings

import numpy as np
import os, sys
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
#sys.path.insert(0, '../../lenstronomy/')


def import_image(file_path, band=1, **kwargs):
    '''
    import fits file images and return numpy array
    return: 2d numpy array
    '''
    hdul = fits.open(file_path)
    lens_gal = hdul['COADD'].data[band]

    return lens_gal

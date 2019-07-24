#initial data settings

import numpy as np
import os, sys
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
sys.path.insert(0, '../../lenstronomy/')

import lenstronomy.Util.data_util as data_util
import lenstronomy.Util.util as util
import lenstronomy.Plots.plot_util as plot_util



def import_image(file_path, band=1, **kwargs):
    '''
    import fits file images and return numpy array
    return: 2d numpy array
    '''
    hdul = fits.open(cutouts_fits_path + file)
    lens_gal = hdul['COADD'].data[band]

    return lens_gal

if __name__ == '__main__':
    cutouts_fits_path = "/Users/yao-yulin/Downloads/cutouts_v2/"
    file = os.listdir(cutouts_fits_path)[0]
    lens_gal = import_image(cutouts_fits_path + file)
    print(lens_gal.shape)
    plt.imshow(lens_gal[:, :])
    plt.show()


#cutouts_fits_path = "/Users/yao-yulin/Downloads/cutouts_v2/"


# files = os.listdir(cutouts_fits_path)
#
# for i, file in enumerate(files):
#     print(file)
#     hdul = fits.open(cutouts_fits_path + file)
#     print(hdul.info())
#     print(hdul['COADD'].data.shape)
#     lens_gal_g = hdul['COADD'].data[1]
#     lens_gal_r = hdul['COADD'].data[2]
#     lens_gal_i = hdul['COADD'].data[3]
#     lens_gal_z = hdul['COADD'].data[4]
#     lens_gal_deg = hdul['COADD'].data[0]
#
#     plt.subplot(1, 5, 1)
#     plt.imshow(np.log10(lens_gal_g))
#
#     plt.subplot(1, 5, 2)
#     plt.imshow(lens_gal_r)
#
#     plt.subplot(1, 5, 3)
#     plt.imshow(lens_gal_i)
#
#     plt.subplot(1, 5, 4)
#     plt.imshow(lens_gal_z)
#
#     plt.subplot(1, 5, 5)
#     plt.imshow(lens_gal_deg)
#
#     plt.show()
#
#
#
#
#
#     #image_data = fits.getdata(hdul, ext=0)
#
#     if i > 2:
#         break

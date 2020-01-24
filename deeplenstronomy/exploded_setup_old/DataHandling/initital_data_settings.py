#initial data settings
from astropy.io import fits


def import_image(file_path, band=1, **kwargs):
    '''
    import fits file images and return numpy array
    return: 2d numpy array
    '''
    hdul = fits.open(file_path)
    lens_gal = hdul['COADD'].data[band]

    return lens_gal

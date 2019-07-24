import os
import pytest
from deeplenstronomy.DataHandling.initital_data_settings import *
#from deeplenstronomy import initial_data_settings
#import deeplenstronomy.initial_data_settings

class TestImportData(object):

    def setup(self):
        pass

    def test_import_image(self):

        cutouts_fits_path = "/test_ImSim/test_files/balrog_cutouts_00102.fits"
        #file = os.listdir(cutouts_fits_path)
        lens_gal = import_image(cutouts_fits_path)
        assert lens_gal.shape == (128, 128)


if __name__ == '__main__':
    pytest.main()

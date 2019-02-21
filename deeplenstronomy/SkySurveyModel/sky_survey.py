import numpy as np


class SurveyBase(object):
    """
    basic access points to a SkySurvey class
    """
    def __init__(self):
        pass


def survey_constructor(survey_name, band_name):

    """

    :param survey_name: name of survey, supported are 'LSST'
    """
    if survey_name == 'LSST':
        return LSST(band_name=band_name)
    else:
        raise ValueError("survey_name %s not supported!")


class LSST(SurveyBase):

    def __init__(self, band_name='g'):
        self.pixelsize = 0.263
        self.bands = np.array(['g', 'r', 'i'])
        if band_name in self.bands:
            self._index_band = int(np.where(self.bands == band_name)[0])
        else:
            raise ValueError("image_band with name %s not found!")
        self.zeropoints = [30, 30, 30]
        self.zeroexposuretime = 90.
        self.skybrightnesses = [21.7, 20.7, 20.1]
        self.exposuretimes = [900, 900, 900]
        self.gains = [4.5, 4.5, 4.5]
        self._seeing = [.9, .9, .9]
        self.nexposures = 10
        self.degrees_of_survey = 5000
        self.readnoise = (10 / 4.5)
        self.psf_type = 'GAUSSIAN'
        self._psf_model_list = [None, None, None]
        super(LSST, self).__init__()

    @property
    def magnitude_zero_point(self):
        return self.zeropoints[self._index_band]

    @property
    def seeing(self):
        return self._seeing[self._index_band]

    @property
    def psf_model(self):
        return self._psf_model_list[self._index_band]

    @property
    def sigma_bkg(self):
        #TODO this needs to be computed based on the quantities above
        return self.readnoise

    @property
    def exposure_time(self):
        return self.exposuretimes[self._index_band]

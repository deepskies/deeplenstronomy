class SkySurveyModel(object):
    """
    These functions define the sky survey parameters for image quality
    """


    def LSST(self):
        self.pixelsize = 0.263
        self.side = 76
        self.bands = ['g', 'r', 'i']
        self.zeropoints = [30, 30, 30]
        self.zeroexposuretime = 90.
        self.skybrightnesses = [21.7, 20.7, 20.1]
        self.exposuretimes = [900, 900, 900]
        self.gains = [4.5, 4.5, 4.5]
        self.seeing = [.9, .9, .9]
        self.nexposures = 10
        self.degrees_of_survey = 5000
        self.readnoise = (10 / 4.5)
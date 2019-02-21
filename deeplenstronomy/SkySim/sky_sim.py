from deeplenstronomy.ImSim.lenstronomy_wrapper import LenstronomyAPI
import numpy as np


class SkySim(object):
    """
    This class simulates a sky, provided a PopulationModule, SkySurveyModule and potentially a selection criteria

    There are two main routines. One is a catalogue based model of lensing properties, and one is at the image level
    level
    """
    def __init__(self, lensPop, skySurvey, cosmo=None):
        """

        :param lensPop: instance of lensPop module
        :param skySurvey: instance of SkySurveyModule
        """
        if cosmo is None:
            from astropy.cosmology import default_cosmology
            cosmo = default_cosmology.get()
        self._lensPop = lensPop
        self._skySurvey = skySurvey
        self._imSim = LenstronomyAPI(self._skySurvey, cosmo=cosmo)

    def lens_cat(self, seed=41):
        """

        :param seed: seed for the realization
        :return: catalogue (list) of lenses
        """
        return self._lensPop.lens_cat(area=self._skySurvey.area, seed=seed)

    def draw_lens_systems(self, num=1, cutout_size=52, seed=41):
        """
        draws a fixed number of lenses

        :param num: number of lenses
        :param seed:
        :return: simulated images
        """
        np.random.seed(seed)
        image_list = []
        for i in range(num):
            z_lens, z_source, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps = self._lensPop.draw_lens_system()
            image = self._imSim.sim_image(numpix=cutout_size, z_lens=z_lens, z_source=z_source, kwargs_lens=kwargs_lens,
                                          kwargs_source=kwargs_source, kwargs_lens_light=kwargs_lens_light,
                                          kwargs_ps=kwargs_ps)
            image_list.append(image)
        return image_list
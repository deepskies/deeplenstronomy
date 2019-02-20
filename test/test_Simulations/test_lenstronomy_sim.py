from deeplenstronomy.Simulations.lenstronomy_sim import LenstronomySim
import numpy as np


class TestLenstronomySim(object):
    """
    tests the class lenstronomySim
    """
    def setup(self):
        pass

    def test_simulate_sis_sersic(self):

        lenstronomySim = LenstronomySim()
        kwargs_lens = [{'theta_E': 1, 'center_x': 0, 'center_y': 0}]
        kwargs_source = [{'amp': 1, 'R_sersic': 1, 'n_sersic': 1, 'center_x': 0.1, 'center_y': 0}]
        kwargs_lens_light = [{'amp': 10, 'R_sersic': 2, 'n_sersic': 4, 'center_x': 0., 'center_y': 0}]
        numpix = 10
        pixelscale = 0.3
        model = lenstronomySim.simulate_sis_sersic(kwargs_lens, kwargs_source, kwargs_lens_light, numpix, pixelscale, psf_image=None)
        nx, ny = np.shape(model)
        assert nx == numpix


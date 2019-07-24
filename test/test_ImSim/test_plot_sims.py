from deeplenstronomy.ImSim import image_sim
from deeplenstronomy.ImSim import plot_sim
from deeplenstronomy.PopSim.population import Population
import pytest
import numpy as np
import matplotlib.pyplot as plt
import numpy.testing as npt


class TestPlotSim(object):

    def setup(self):
        pass

    def test_image_sim(self):
        pop = Population()
        kwargs_params, kwargs_model = pop.draw_model(with_lens_light=True, with_quasar=True)

        kwargs_band = {'read_noise': 10,
                    'pixel_scale': 0.263,
                    'ccd_gain': 4.5,
                       'exposure_time': 90.,
                           'magnitude_zero_point': 30,
                           'num_exposures': 10,
                           'psf_type': 'GAUSSIAN',
                       'seeing': 1.0,
                       'sky_brightness': 21}
        numpix = 10
        # print(kwargs_params['kwargs_lens'])
        image = image_sim.sim_image(numpix, kwargs_band, kwargs_model, kwargs_params, kwargs_numerics={})
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        plot_sim.plot_single_band(ax, image)
        plt.close()


if __name__ == '__main__':
    pytest.main()
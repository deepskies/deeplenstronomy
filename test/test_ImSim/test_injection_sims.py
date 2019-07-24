from deeplenstronomy.ImSim import inject_simulations
from deeplenstronomy.PopSim.population import Population
import pytest
import numpy as np
import numpy.testing as npt


class TestPopulation(object):

    def setup(self):
        pass

    def test_add_arc(self):
        pop = Population()
        kwargs_params, kwargs_model = pop.draw_model(with_lens_light=True, with_quasar=True)
        image = np.zeros((10, 10))

        kwargs_band = {'read_noise': 10,
                    'pixel_scale': 0.263,
                    'ccd_gain': 4.5,
                       'exposure_time': 90.,
                           'magnitude_zero_point': 30,
                           'num_exposures': 10,
                           'psf_type': 'GAUSSIAN',
                       'seeing': 1.0,
                       'sky_brightness': 21}

        added_image = inject_simulations.add_arc(image, kwargs_band, kwargs_params, kwargs_model, kwargs_numerics={})
        assert np.sum(added_image) > 0



if __name__ == '__main__':
    pytest.main()

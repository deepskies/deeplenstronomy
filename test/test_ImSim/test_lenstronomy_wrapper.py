from deeplenstronomy.ImSim.lenstronomy_wrapper import LenstronomyAPI
import numpy.testing as npt
import numpy as np


class TestLenstronomyWrapper(object):
    """
    tests the class lenstronomySim
    """
    def setup(self):
        class DataInstance(object):
            def __init__(self):
                pass
        data_instance = DataInstance()
        data_instance.pixelsize = 0.13
        data_instance.magnitude_zero_point = 20
        data_instance.psf_type = 'GAUSSIAN'
        data_instance.seeing = 0.93
        data_instance.psf_model = None
        data_instance.sigma_bkg = 0.1
        data_instance.exposure_time = 900
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
        self.sim = LenstronomyAPI(skySurvey=data_instance, cosmo=cosmo)

    def test_sim_image(self):
        numpix = 76
        z_lens, z_source = 0.5, 2.
        mass = 0
        velocity_dispersion = 250
        axis_ratio_lens, inclination_angle_lens = 0.8, 0.3
        lens_center_ra, lens_center_dec = 0, 0
        magnitude_lens_light, halflight_radius_lens_light, n_sersic_lens_light = 15, 1., 4
        axis_ratio_lens_light, inclination_angle_lens_light = 0.9, 0.3
        lens_light_center_ra, lens_light_center_dec = 0, 0
        magnitude_source, halflight_radius_source, n_sersic_source = 17, 0.2, 1
        axis_ratio_source, inclination_angle_source = 0.7, -0.3
        source_center_ra, source_center_dec = 0.1, 0.

        kwargs_lens = {'velocity_dispersion': velocity_dispersion, 'axis_ratio': axis_ratio_lens,
                       'inclination_angle': inclination_angle_lens, 'center_ra': 0, 'center_dec': 0}
        kwargs_source = {'magnitude': magnitude_source, 'halflight_radius': halflight_radius_source,
                         'n_sersic': n_sersic_source, 'axis_ratio': axis_ratio_source,
                         'inclination_angle': inclination_angle_source, 'center_ra': source_center_ra,
                         'center_dec': source_center_dec}
        kwargs_lens_light = {'magnitude': magnitude_lens_light, 'halflight_radius': halflight_radius_lens_light,
                             'n_sersic': n_sersic_lens_light, 'axis_ratio': axis_ratio_lens_light,
                             'inclination_angle': inclination_angle_lens_light, 'center_ra': lens_light_center_dec,
                             'center_dec': lens_light_center_dec}
        kwargs_ps = None
        model = self.sim.sim_image(numpix, z_lens, z_source, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
        npt.assert_almost_equal(np.sum(model), 477.81985038336063, decimal=-3)




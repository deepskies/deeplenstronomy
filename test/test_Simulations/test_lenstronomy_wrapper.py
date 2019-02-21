from deeplenstronomy.ImSim.lenstronomy_wrapper import LenstronomyAPI
import numpy.testing as npt
import numpy as np


class TestSimPhys2Image(object):
    """
    tests the class lenstronomySim
    """
    def setup(self):
        class DataInstance(object):
            def __init__(self):
                pass
        data_instance = DataInstance()
        data_instance.pixelscale = 0.13
        data_instance.magnitude_zero_point = 20
        data_instance.psf_type = 'GAUSSIAN'
        data_instance.psf_fwhm = 0.93
        data_instance.psf_model = None
        data_instance.sigma_bkg = 0.1
        data_instance.exposure_time = 900
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
        self.sim = LenstronomyAPI(data_instance=data_instance, numpix=76, cosmo=cosmo)

    def test_sim_image(self):
        z_lens, z_source = 0.5, 2.
        velocity_dispersion = 250
        axis_ratio_lens, inclination_angle_lens = 0.8, 0.3
        lens_center_ra, lens_center_dec = 0, 0
        magnitude_lens_light, halflight_radius_lens_light, n_sersic_lens_light = 15, 1., 4
        axis_ratio_lens_light, inclination_angle_lens_light = 0.9, 0.3
        lens_light_center_ra, lens_light_center_dec = 0, 0
        magnitude_source, halflight_radius_source, n_sersic_source = 17, 0.2, 1
        axis_ratio_source, inclination_angle_source = 0.7, -0.3
        source_center_ra, source_center_dec = 0.1, 0.
        model = self.sim.sim_image(z_lens, z_source, velocity_dispersion, axis_ratio_lens, inclination_angle_lens, lens_center_ra,
                  lens_center_dec, magnitude_lens_light, halflight_radius_lens_light, n_sersic_lens_light,
                  axis_ratio_lens_light, inclination_angle_lens_light, lens_light_center_ra, lens_light_center_dec,
                  magnitude_source, halflight_radius_source, n_sersic_source, axis_ratio_source,
                  inclination_angle_source, source_center_ra, source_center_dec)
        npt.assert_almost_equal(np.sum(model), 477.81985038336063, decimal=-3)




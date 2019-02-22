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
        data_instance.sigma_bkg = 0.01
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

    def test_image_injections(self):
        import lenstronomy.Util.util as util
        x_grid, y_grid = util.make_grid(numPix=20, deltapix=0.4)
        from lenstronomy.LightModel.light_model import LightModel
        lightModel = LightModel(light_model_list=['SERSIC'])
        magnitude_lens = 10
        magnitude_source = 13
        kwargs_lens_light_init = [{'amp': 10, 'R_sersic': 1, 'n_sersic': 3, 'center_x': 0, 'center_y': 0}]
        image_lens = lightModel.surface_brightness(x_grid, y_grid, kwargs_lens_light_init)
        image_lens = util.array2image(image_lens)
        kwargs_lens_light = {'image': image_lens, 'pixelsize': 0.1, 'magnitude': magnitude_lens, 'relative_rotation': 0,
                             'center_ra': 0, 'center_dec':0}
        kwargs_lens_light_analytic = {'magnitude': magnitude_lens, 'halflight_radius': 1./4, 'n_sersic': 3, 'center_ra': 0, 'center_dec': 0,
                                      'axis_ratio': 1, 'inclination_angle': 0}
        kwargs_source = {'image': image_lens, 'pixelsize': 0.05, 'magnitude': magnitude_source, 'relative_rotation': 0,
                             'center_ra': 0.2, 'center_dec': 0}
        kwargs_source_analytic = {'magnitude': magnitude_source, 'halflight_radius': 1. / 8, 'n_sersic': 3, 'center_ra': 0.2, 'center_dec': 0,
                                      'axis_ratio': 1, 'inclination_angle': 0}

        numpix = 76
        z_lens, z_source = 0.5, 2.
        velocity_dispersion = 250
        axis_ratio_lens, inclination_angle_lens = 0.8, 0.3
        kwargs_lens = {'velocity_dispersion': velocity_dispersion, 'axis_ratio': axis_ratio_lens,
                       'inclination_angle': inclination_angle_lens, 'center_ra': 0, 'center_dec': 0}
        kwargs_ps = None
        model = self.sim.sim_image(numpix, z_lens, z_source, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
        model_analytic = self.sim.sim_image(numpix, z_lens, z_source, kwargs_lens, kwargs_source_analytic,
                                            kwargs_lens_light_analytic, kwargs_ps)
        npt.assert_almost_equal(np.sum(model) / np.sum(model_analytic), 1, decimal=1)

    def test_magnitude_definition(self):
        mag_zero_point = 10
        pixelsize = 0.13
        class DataInstance(object):
            def __init__(self):
                pass
        data_instance = DataInstance()
        data_instance.pixelsize = pixelsize
        data_instance.magnitude_zero_point = mag_zero_point
        data_instance.psf_type = 'GAUSSIAN'
        data_instance.seeing = 0.93
        data_instance.psf_model = None
        data_instance.sigma_bkg = 0.000001
        data_instance.exposure_time = 90000
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
        self.sim = LenstronomyAPI(skySurvey=data_instance, cosmo=cosmo)


        kwargs_lens_light = {'magnitude': mag_zero_point, 'halflight_radius': 2, 'n_sersic': 3,
                                      'center_ra': 0, 'center_dec': 0,
                                      'axis_ratio': 1, 'inclination_angle': 0}
        numpix = 200
        z_lens, z_source = 0.5, 2
        model = self.sim.sim_image(numpix, z_lens, z_source, kwargs_lens_light=kwargs_lens_light)
        npt.assert_almost_equal(np.sum(model), 1, decimal=-1)

        kwargs_lens_light = {'image': model, 'pixelsize': pixelsize, 'magnitude': mag_zero_point, 'relative_rotation': 0,
                             'center_ra': 0, 'center_dec': 0}
        model_interp = self.sim.sim_image(numpix, z_lens, z_source, kwargs_lens_light=kwargs_lens_light)
        npt.assert_almost_equal(np.sum(model_interp), 1, decimal=-4)

        mag = 10
        kwargs_lens_light = {'magnitude': mag, 'halflight_radius': 2, 'n_sersic': 3,
                                      'center_ra': 0, 'center_dec': 0,
                                      'axis_ratio': 1, 'inclination_angle': 0}
        numpix = 200
        z_lens, z_source = 0.5, 2
        model = self.sim.sim_image(numpix, z_lens, z_source, kwargs_lens_light=kwargs_lens_light)

        kwargs_lens_light = {'image': model, 'pixelsize': pixelsize, 'magnitude': mag, 'relative_rotation': 0,
                             'center_ra': 0, 'center_dec': 0}
        model_interp = self.sim.sim_image(numpix, z_lens, z_source, kwargs_lens_light=kwargs_lens_light)
        npt.assert_almost_equal(np.sum(model_interp), np.sum(model), decimal=-4)

    def test_point_source(self):
        kwargs_ps = {'magnitude': 15, 'center_ra': 0.0, 'center_dec': 0.0}
        velocity_dispersion = 350.
        axis_ratio_lens = .7
        inclination_angle_lens=.0
        z_lens = 0.5
        z_source = 1.4
        numpix = 64

        # point source as star as artifact, set source type list to include 'UNLENSED'
        kwargs_lens = {'velocity_dispersion': velocity_dispersion, 'axis_ratio': axis_ratio_lens,
                       'inclination_angle': inclination_angle_lens, 'center_ra': 0, 'center_dec': 0}
        model = self.sim.sim_image(numpix, z_lens, z_source, kwargs_lens=kwargs_lens, kwargs_ps=kwargs_ps)
        #npt.assert_almost_equal(np.sum(model), 10, decimal=-1)

        import matplotlib.pyplot as plt
        plt.matshow(model)
        plt.show()
        assert 1==0


from deeplenstronomy.PopSim.population_model import PopulationModel
import numpy as np
import numpy.testing as npt


class TestPopulationModel(object):
    """
    tests the class PopSim
    """


    def setup(self):
        np.random.seed(921)
        pass

    def test_draw_mass(self):
        """
        test the draw mass function
        :return:
        """

        population_model = PopulationModel()
        mean = 10
        sigma = 1
        mass = population_model.DrawHaloMass(mean, sigma)
        npt.assert_almost_equal(mass, 10.096290999771787, decimal=3)


    def test_draw_redshift(self):
        """
        test the draw redshift function
        :return:
        """

        population_model = PopulationModel()
        mean = 0.5
        sigma = 0.1
        redshift = population_model.DrawRedshift(mean,sigma)
        npt.assert_almost_equal(redshift, 0.5096290999771788, decimal=3)



    def test_draw_velocity_dispersion(self):
        """
        test the draw redshift function
        :return:
        """

        population_model = PopulationModel()
        mean = 300
        sigma = 20
        velocity_dispersion = population_model.DrawVelocityDispersion(mean, sigma)
        npt.assert_almost_equal(velocity_dispersion, 301.92581999543575, decimal=3)

    def test_draw_position(self):
        """
        test the draw position function
        :return:
        """

        population_model = PopulationModel()
        ra_min, ra_max = 0, 100
        dec_min, dec_max = 0, 100
        ra, dec = population_model.DrawPosition(ra_min, ra_max, dec_min, dec_max)
        npt.assert_almost_equal(ra, 96.56181023309233 , decimal=3)
        npt.assert_almost_equal(dec, 29.582799280733994, decimal=3)

    def test_draw_axis_ratio(self):
        """
        test the draw axis ratio function
        :return:
        """

        population_model = PopulationModel()
        mean = 0.8
        sigma = 0.1
        axisratio = population_model.DrawAxisRatio(mean, sigma)
        npt.assert_almost_equal(axisratio, 0.8096290999771788, decimal=3)

    def test_draw_inclination_angle(self):
        """
        test the draw inclination angle function
        :return:
        """

        population_model = PopulationModel()
        mean = 0.3
        sigma = 0.05
        inclination_angle = population_model.DrawInclinationAngle(mean, sigma)
        npt.assert_almost_equal(inclination_angle, 0.30481454998858937, decimal=3)

    def test_draw_magnitude(self):
        """
        test the draw magnitude function
        :return:
        """

        population_model = PopulationModel()
        mean = 20
        sigma = 1
        magnitude = population_model.DrawMagnitude(mean, sigma)
        npt.assert_almost_equal(magnitude, 20.09629099977179, decimal=3)

    def test_draw_half_light_radius(self):
        """
        test the draw half light radius function
        :return:
        """

        population_model = PopulationModel()
        mean = 1.0
        sigma = 0.1
        half_light_radius = population_model.DrawVelocityDispersion(mean, sigma)
        npt.assert_almost_equal(half_light_radius, 1.0096290999771786, decimal=3)

    def test_draw_n_sersic(self):
        """
        test the draw n_sersic function
        :return:
        """

        population_model = PopulationModel()
        mean = 4
        sigma = 0.2
        n_sersic = population_model.DrawVelocityDispersion(mean, sigma)
        npt.assert_almost_equal(n_sersic, 4.019258199954358, decimal=3)

    def test_draw_all(self):
        """
        test draw all the variables
        :return:
        """

        population_model = PopulationModel()
        kwargs = population_model.draw_lens_system()

        npt.assert_almost_equal(kwargs['mass'], 10.096290999771787, decimal=3)
        #npt.assert_almost_equal(redshift_lens, 0.4551340730445048, decimal=3)
        #npt.assert_almost_equal(redshift_source, 0.40647048740880787, decimal=3)
        #npt.assert_almost_equal(velocity_dispersion, 323.3514562472069, decimal=3)
        #npt.assert_almost_equal(axis_ratio_lens, 0.8936058849251, decimal=3)
        #npt.assert_almost_equal(axis_ratio_lens_light, 0.8660626001523951, decimal=3)
        #npt.assert_almost_equal(axis_ratio_source, 0.7734542046198142, decimal=3)
        #npt.assert_almost_equal(inclination_angle_lens, 0.3233175589928309, decimal=3)
        #npt.assert_almost_equal(inclination_angle_lens_light, 0.37916208538579854, decimal=3)
        #npt.assert_almost_equal(inclination_angle_source, -0.3458299701587175, decimal=3)
        #npt.assert_almost_equal(ra_lens_light_center, 12.594734408326058, decimal=3)
        #npt.assert_almost_equal(dec_lens_light_center, 33.45680779907787, decimal=3)
        #npt.assert_almost_equal(ra_source_center, 32.43179784221102, decimal=3)
        #npt.assert_almost_equal(dec_source_center, 27.35023094338922, decimal=3)
        #npt.assert_almost_equal(magnitude_lens_light, 19.333401875315918, decimal=3)
        #npt.assert_almost_equal(magnitude_source, 17.711730766532412, decimal=3)
        #npt.assert_almost_equal(halflight_radius_lens_light, 1.2035913147214066, decimal=3)
        #npt.assert_almost_equal(halflight_radius_source, 1.0096290999771786, decimal=3)
        #npt.assert_almost_equal(n_sersic_lens_light, 3.6163411977935547, decimal=3)
        #npt.assert_almost_equal(n_sersic_source, 3.864353321588361, decimal=3)
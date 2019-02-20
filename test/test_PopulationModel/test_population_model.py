from deeplenstronomy.PopulationModel.lenstronomy_popmodel import PopulationModel
import numpy as np


class TestPopulationModel(object):
    """
    tests the class PopulationModel
    """
    def setup(self):
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
        assert mass < mean + 10*sigma, "Mass out of bounds; too high"
        assert mass > mean - 10*sigma, "Mass out of bounds; too low"


    def test_draw_redshift(self):
        """
        test the draw redshift function
        :return:
        """

        population_model = PopulationModel()
        mean = 0.5
        sigma = 0.1
        redshift = population_model.DrawRedshift(mean,sigma)
        assert redshift < mean + 10*sigma, "redshift out of bounds; too high"
        assert redshift > mean - 10*sigma, "redshift out of bounds; too low"


    def test_draw_velocity_dispersion(self):
        """
        test the draw redshift function
        :return:
        """

        population_model = PopulationModel()
        mean = 0.5
        sigma = 0.1
        velocity_dispersion = population_model.DrawVelocityDispersion(mean, sigma)
        assert velocity_dispersion < mean + 10 * sigma, "velocity disp out of bounds; too high"
        assert velocity_dispersion > mean - 10 * sigma, "velocity disp out of bounds; too low"


    def test_draw_source_position(self):
        """
        test the draw redshift function
        :return:
        """

        population_model = PopulationModel()
        ra_min, ra_max = 0, 10
        dec_min, dec_max = 0, 10
        ra, dec = population_model.DrawSourcePosition(ra_min, ra_max, dec_min, dec_max)
        assert ra_min < ra < ra_max, "ra out of bounds"
        assert dec_min < dec < dec_max, "dec out of bounds"
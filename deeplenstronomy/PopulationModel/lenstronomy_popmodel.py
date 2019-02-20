from numpy import random


class PopulationModel(object):
    """
    This class defines the population draws needed to produce a large set of lensing systems
    """

    def DrawHaloMass(self, mean, sigma):
        """
        Defines mass function for galaxy-scale lenses
        :return:
        """
        return random.normal(loc=mean, scale=sigma)


    def DrawRedshift(self, mean=0.5, sigma=0.1):
        """
        Draws position angle from a distribution
        :return:
        """
        return random.normal(loc=mean, scale=sigma)
        return redshift


    def DrawVelocityDispersion(self, mean=300, sigma=50):
        """
        Draw Velocity Dispersion from a distribution
        :param mean:
        :param sigma:
        :return:
        """


    def DrawSourcePosition(self, ra_min, ra_max, dec_min, dec_max):
        """
        Draw source position from a distribution
        :param dec_max:
        :return:
        """
        ra = random.uniform(low=ra_min, high=ra_max)
        dec = random.uniform(low=dec_min, high=dec_max)
        return ra, dec








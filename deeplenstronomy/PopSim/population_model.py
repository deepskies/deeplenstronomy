from numpy import random


class PopulationModel(object):
    """
    This class defines the population draws needed to produce a large set of lensing systems
    """

    #def __int__(self, seed=None, kwargs=None):
    #    self.mass_mean = kwargs.get('mass_mean')
    #    self

    def __init__(self):
        """
        Constructor for Population Model
        """
        self.mass_mean=10
        self.mass_sigma=1
        self.redshift_lens_mean=0.5
        self.redshift_lens_sigma=0.1
        self.redshift_source_mean=0.5
        self.redshift_source_sigma=0.1
        self.velocity_dispersion_mean=300.
        self.velocity_dispersion_sigma=20.

        self.axis_ratio_lens_mean=0.8
        self.axis_ratio_lens_sigma=0.1
        self.axis_ratio_lens_light_mean = 0.8
        self.axis_ratio_lens_light_sigma = 0.1
        self.axis_ratio_source_mean = 0.8
        self.axis_ratio_source_sigma = 0.1

        self.inclination_angle_lens_mean=0.3
        self.inclination_angle_lens_sigma=0.05
        self.inclination_angle_lens_light_mean=0.3
        self.inclination_angle_lens_light_sigma=0.05
        self.inclination_angle_source_mean=-0.3
        self.inclination_angle_source_sigma=0.05

        self.lens_light_center_ra_min=0.0
        self.lens_light_center_ra_max=100
        self.lens_light_center_dec_min=0.0
        self.lens_light_center_dec_max=100
        self.source_center_ra_min=0
        self.source_center_ra_max=100
        self.source_center_dec_min=0
        self.source_center_dec_max=100

        self.magnitude_lens_light_mean=20.
        self.magnitude_lens_light_sigma=1.
        self.magnitude_source_mean = 20.
        self.magnitude_source_sigma = 1.

        self.halflight_radius_lens_light_mean=1.
        self.halflight_radius_lens_light_sigma=0.1
        self.halflight_radius_source_mean=1.0
        self.halflight_radius_source_sigma=0.1

        self.n_sersic_lens_light_mean=4
        self.n_sersic_lens_light_sigma=0.2
        self.n_sersic_source_mean = 4.
        self.n_sersic_source_sigma = 0.2

    def draw_lens_system(self):
        """
        Draw all of the relevant variables
        :return:
        """
        mass = self.DrawHaloMass(self.mass_mean, self.mass_sigma)
        z_lens = self.DrawRedshift(self.redshift_lens_mean, self.redshift_lens_sigma)
        z_source = self.DrawRedshift(self.redshift_source_mean, self.redshift_source_sigma)
        velocity_dispersion = self.DrawVelocityDispersion(self.velocity_dispersion_mean, self.velocity_dispersion_sigma)

        axis_ratio_lens_light = self.DrawAxisRatio(self.axis_ratio_lens_light_mean, self.axis_ratio_lens_light_sigma)
        axis_ratio_lens = self.DrawAxisRatio(self.axis_ratio_lens_mean, self.axis_ratio_lens_sigma)
        axis_ratio_source = self.DrawAxisRatio(self.axis_ratio_source_mean, self.axis_ratio_source_sigma)

        inclination_angle_lens = self.DrawInclinationAngle(self.inclination_angle_lens_mean,
                                                           self.inclination_angle_lens_sigma)
        inclination_angle_lens_light = self.DrawInclinationAngle(self.inclination_angle_lens_light_mean,
                                                           self.inclination_angle_lens_light_sigma)
        inclination_angle_source = self.DrawInclinationAngle(self.inclination_angle_source_mean,
                                                             self.inclination_angle_source_sigma)

        ra_lens_light_center, dec_lens_light_center = self.DrawPosition(self.lens_light_center_ra_min,
                                                                         self.lens_light_center_ra_max,
                                                                         self.lens_light_center_dec_min,
                                                                         self.lens_light_center_dec_max)
        ra_source_center, dec_source_center = self.DrawPosition(self.source_center_ra_min, self.source_center_ra_max,
                                                                self.source_center_dec_min, self.source_center_dec_max)

        magnitude_lens_light = self.DrawMagnitude(self.magnitude_lens_light_mean, self.magnitude_lens_light_sigma)
        magnitude_source = self.DrawMagnitude(self.magnitude_source_mean, self.magnitude_source_sigma)
        halflight_radius_lens_light = self.DrawHalfLightRadius(self.halflight_radius_lens_light_mean,
                                                                self.halflight_radius_lens_light_sigma)
        halflight_radius_source = self.DrawHalfLightRadius(self.halflight_radius_source_mean,
                                                           self.halflight_radius_source_sigma)

        n_sersic_lens_light = self.DrawNSersic(self.n_sersic_lens_light_mean, self.n_sersic_lens_light_sigma)
        n_sersic_source = self.DrawNSersic(self.n_sersic_source_mean, self.n_sersic_source_sigma)
        kwargs_lens = {'velocity_dispersion': velocity_dispersion, 'axis_ratio': axis_ratio_lens,
                       'inclination_angle': inclination_angle_lens, 'center_ra': 0, 'center_dec': 0}
        kwargs_source = {'magnitude': magnitude_source, 'halflight_radius': halflight_radius_source,
                         'n_sersic': n_sersic_source, 'axis_ratio': axis_ratio_source,
                         'inclination_angle': inclination_angle_source, 'center_ra': ra_source_center,
                         'center_dec': dec_source_center}
        kwargs_lens_light = {'magnitude': magnitude_lens_light, 'halflight_radius': halflight_radius_lens_light,
                         'n_sersic': n_sersic_lens_light, 'axis_ratio': axis_ratio_lens_light,
                         'inclination_angle': inclination_angle_lens_light, 'center_ra': ra_lens_light_center,
                         'center_dec': dec_lens_light_center}
        kwargs_ps = None
        return z_lens, z_source, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps

    def DrawAxisRatio(self, mean, sigma):
        """
        Draw axis ratio
        :return:
        """
        return random.normal(loc=mean, scale=sigma)

    def DrawInclinationAngle(self, mean, sigma):
        """
        Draw Inclination angle
        :param mean:
        :param sigma:
        :return:
        """
        return random.normal(loc=mean, scale=sigma)

    def DrawMagnitude(self, mean, sigma):
        """
        Draw magnitude
        :param mean:
        :param sigma:
        :return:
        """
        return random.normal(loc=mean, scale=sigma)

    def DrawHalfLightRadius(self, mean, sigma):
        """
        Draw half light radius
        :param mean:
        :param sigma:
        :return:
        """
        return random.normal(loc=mean, scale=sigma)

    def DrawNSersic(self, mean, sigma):
        """
        Draw N sersic index
        :param mean:
        :param sigma:
        :return:
        """
        return random.normal(loc=mean, scale=sigma)


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


    def DrawVelocityDispersion(self, mean=300, sigma=50):
        """
        Draw Velocity Dispersion from a distribution
        :param mean:
        :param sigma:
        :return:
        """
        return random.normal(loc=mean, scale=sigma)

    def DrawPosition(self, ra_min, ra_max, dec_min, dec_max):
        """
        Draw  position from a distribution
        :param dec_max:
        :return:
        """
        ra = random.uniform(low=ra_min, high=ra_max)
        dec = random.uniform(low=dec_min, high=dec_max)
        return ra, dec









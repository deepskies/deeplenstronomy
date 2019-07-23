class GalaxyProp(object):
    """
    This class manages all the galaxy properties at all redshift

    """
    def __init__(self):
        pass

    def draw_cone(self, width, z_min, z_max, mag_min):
        """
        draws all galaxies lying in a cone of width

        :param width: width inb arcseconds
        :param z_min: minimum redshift to draw from
        :param z_max: maximum redshift to draw from
        :param mag_min: minimum magnitude to draw from
        :return: list of objects being present in the cone
        """
        return 0

    def draw_ligth2_mass(self):
        pass


class SourcePop():

    def __int__(self):
        pass

    def draw_source_model():
        """
        draws source model from population
        """
        source_center_x = (np.random.rand() - 0.5) * 2
        source_center_y = (np.random.rand() - 0.5) * 2
        kwargs_source_mag = [{'magnitude': 22, 'R_sersic': 0.3, 'n_sersic': 1, 'e1': -0.3, 'e2': -0.2, 'center_x': source_center_x, 'center_y': source_center_y}]
        source_model_list = ['SERSIC_ELLPISE']
        return kwargs_source_mag, source_model_list

    def draw_lens_model():
        """
        draw lens model parameters
        return: lens model keyword argument list, lens model list
        """
        theta_E = np.random.uniform(0.9, 2.2)
        lens_e1 = (np.random.rand() - 0.5) * 0.8
        lens_e2 = (np.random.rand() - 0.5) * 0.8
        kwargs_lens = [
        {'theta_E': theta_E, 'e1': lens_e1, 'e2': lens_e2, 'center_x': 0, 'center_y': 0},  # SIE model
        {'e1': 0.03, 'e2': 0.01}  # SHEAR model
        ]
        lens_model_list = ['SIE', 'SHEAR']
        return kwargs_lens, lens_model_list

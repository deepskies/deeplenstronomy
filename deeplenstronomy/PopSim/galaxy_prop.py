import numpy as np


class SourcePop():

    def __int__(self):
        pass

    def draw_source_model(self):
        """
        draws source model from population
        """
        source_center_x = (np.random.rand() - 0.5) * 2
        source_center_y = (np.random.rand() - 0.5) * 2
        kwargs_source_mag = [{'magnitude': 22, 'R_sersic': 0.3, 'n_sersic': 1, 'e1': -0.3, 'e2': -0.2, 'center_x': source_center_x, 'center_y': source_center_y}]
        source_model_list = ['SERSIC_ELLPISE']
        return kwargs_source_mag, source_model_list

    def draw_lens_model(self):
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

    def draw_lens_light(self):
        """

        :return:
        """
        lens_light_model_list = ['SERSIC)_ELLIPSE']
        kwargs_lens_light = [{'magnitude': 22, 'R_sersic': 0.3, 'n_sersic': 1, 'e1': -0.3, 'e2': -0.2, 'center_x': 0, 'center_y': 0}]
        return kwargs_lens_light, lens_light_model_list

    def draw_point_source(self, center_x, center_y):
        """

        :param center_x: center of point source in source plane
        :param center_y: center of point source in source plane
        :return:
        """
        point_source_model_list = ['SOURCE_POSITION']
        kwargs_ps = [{'magnitude': 21, 'ra_source': center_x, 'dec_source': center_y}]
        return kwargs_ps, point_source_model_list

    def draw_model(self, with_lens_light=False, with_quasar=False, **kwargs):
        """

        :param kwargs:
        :return: kwargs_params, kwargs_model
        """
        kwargs_lens, lens_model_list = self.draw_lens_model()
        kwargs_params = {'kwargs_lens': kwargs_lens}
        kwargs_model = {'lens_model_list': lens_model_list}
        kwargs_source, source_model_list = self.draw_source_model()
        kwargs_params['kwargs_source'] = kwargs_source
        kwargs_model['source_model_list'] = source_model_list
        if with_lens_light:
            kwargs_lens_light, lens_light_model_list = self.draw_lens_light()
            kwargs_params['kwargs_lens_light'] = kwargs_lens_light
            kwargs_model['lens_light_model_list'] = lens_light_model_list
        if with_quasar:
            kwargs_ps, point_source_model_list = self.draw_point_source(center_x=kwargs_source[0]['center_x'], center_y=kwargs_source[0]['center_y'])
            kwargs_params['kwargs_ps'] = kwargs_ps
            kwargs_model['point_source_model_list'] = point_source_model_list
        return kwargs_params, kwargs_model

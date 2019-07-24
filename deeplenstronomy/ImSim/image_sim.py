from lenstronomy.SimulationAPI.sim_api import SimAPI


def sim_image(numpix, kwargs_band, kwargs_model, kwargs_params, kwargs_numerics={}):
    """

    :param numpix:
    :param kwargs_band:
    :param kwargs_model:
    :param kwargs_params:
    :param kwargs_numerics:
    :return:
    """
    #TODO documentation
    sim = SimAPI(numpix=numpix, kwargs_single_band=kwargs_band, kwargs_model=kwargs_model,
                 kwargs_numerics=kwargs_numerics)
    kwargs_lens_light_mag = kwargs_params.get('kwargs_lens_light_mag', None)
    kwargs_source_mag = kwargs_params.get('kwargs_source_mag', None)
    kwargs_ps_mag = kwargs_params.get('kwargs_ps_mag', None)
    kwargs_lens_light, kwargs_source, kwargs_ps = sim.magnitude2amplitude(kwargs_lens_light_mag, kwargs_source_mag, kwargs_ps_mag)
    imSim = sim.image_model_class
    image = imSim.image(kwargs_params['kwargs_lens'], kwargs_lens_light, kwargs_source, kwargs_ps)
    image += sim.noise_for_model(model=image)
    return image
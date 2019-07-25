from lenstronomy.SimulationAPI.sim_api import SimAPI


def sim_image(numpix, kwargs_band, kwargs_model, kwargs_params, kwargs_numerics={}):
    """
    simulates an image based on chosen model and data settings, effectively makes use of lenstronomy.SimulationAPI

    :param numpix: number of pixels per axis
    :param kwargs_band: keyword arguments specifying the observation to be simulated according to lenstronomy.SimulationAPI
    :param kwargs_model: keyword arguments of model configurations. All possibilities available at lenstronom.Util.class_creator
    :param kwargs_params: keyword arguments of the different model components. Supports 'kwargs_lens', 'kwargs_source_mag',
    'kwargs_lens_light_mag', 'kwargs_ps_mag'
    :param kwargs_numerics: keyword arguments describing the numerical setting of lenstronomy as outlined in lenstronomy.ImSim.Numerics
    :return: 2d numpy array
    """

    sim = SimAPI(numpix=numpix, kwargs_single_band=kwargs_band, kwargs_model=kwargs_model,
                 kwargs_numerics=kwargs_numerics)
    kwargs_lens_light_mag = kwargs_params.get('kwargs_lens_light_mag', None)
    kwargs_source_mag = kwargs_params.get('kwargs_source_mag', None)
    kwargs_ps_mag = kwargs_params.get('kwargs_ps_mag', None)
    kwargs_lens_light, kwargs_source, kwargs_ps = sim.magnitude2amplitude(kwargs_lens_light_mag, kwargs_source_mag, kwargs_ps_mag)
    imSim = sim.image_model_class
    image = imSim.image(kwargs_params['kwargs_lens'], kwargs_source, kwargs_lens_light, kwargs_ps)
    image += sim.noise_for_model(model=image)
    return image

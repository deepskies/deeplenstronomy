from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.SimulationAPI.observation_api import SingleBand


def add_arc(image, kwargs_band, kwargs_params, kwargs_model, kwargs_numerics={}):
    """
    routine to add lensed arc to existing image

    :param image: 2d square numpy array of original image
    :param kwargs_band: keyword arguments specifying the observation to be simulated according to lenstronomy.SimulationAPI
    :param kwargs_model: keyword arguments of model configurations. All possibilities available at lenstronom.Util.class_creator
    :param kwargs_params: keyword arguments of the different model components. Supports 'kwargs_lens', 'kwargs_source_mag',
    'kwargs_lens_light_mag', 'kwargs_ps_mag'
    :param kwargs_numerics: keyword arguments describing the numerical setting of lenstronomy as outlined in lenstronomy.ImSim.Numerics
    :return: 2d numpy array
    """
    numpix = len(image)
    arc = _arc_model(numpix, kwargs_band, kwargs_model, kwargs_numerics=kwargs_numerics, **kwargs_params)
    band = SingleBand(**kwargs_band)
    noisy_arc = arc + band.flux_noise(arc)
    return image + noisy_arc


def _arc_model(numpix, kwargs_band, kwargs_model, kwargs_lens, kwargs_source_mag=None, kwargs_lens_light_mag=None,
               kwargs_ps_mag=None, kwargs_numerics={}):
    """
    routine to simulate a lensing arc, wrapper around lenstronomy SimulationAPI module

    :param numpix: number of pixels per axis
    :param kwargs_band: keyword arguments specifying the observation to be simulated according to lenstronomy.SimulationAPI
    :param kwargs_model: keyword arguments of model configurations. All possibilities available at lenstronom.Util.class_creator
    :param kwargs_lens: list of lens model keyword arguments
    :param kwargs_source_mag: list of extended source model keyword arguments
    :param kwargs_lens_light_mag: list of lens light keyword arguments
    :param kwargs_ps_mag: list of point source keyword arguments
    :param kwargs_numerics: keyword arguments describing the numerical setting of lenstronomy as outlined in lenstronomy.ImSim.Numerics
    :return: 2d numpy array
    """

    simAPI = SimAPI(numpix=numpix, kwargs_single_band=kwargs_band, kwargs_model=kwargs_model,
                    kwargs_numerics=kwargs_numerics)

    imSim = simAPI.image_model_class
    kwargs_lens_light, kwargs_source, kwargs_ps = simAPI.magnitude2amplitude(kwargs_source_mag=kwargs_source_mag,
                                                                             kwargs_lens_light_mag=kwargs_lens_light_mag,
                                                                             kwargs_ps_mag=kwargs_ps_mag)
    image = imSim.image(kwargs_lens, kwargs_source=kwargs_source, kwargs_lens_light=kwargs_lens_light,
                        kwargs_ps=kwargs_ps)
    return image

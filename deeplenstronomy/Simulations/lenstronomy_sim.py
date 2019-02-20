import lenstronomy.Util.util as util
from lenstronomy.Data.imaging_data import Data
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel


class LenstronomySim(object):
    """
    this class handles the lenstronomy simulations
    """

    def simulate_sis_sersic(self, kwargs_lens, kwargs_source, kwargs_lens_light, numpix, pixelscale, psf_image=None):
        """

        :param kwargs_lens: kwargs list of lens according to lenstronomy conventions
        :param kwargs_source: kwargs for source according to lenstronomy conventions
        :param numpix: number of pixels
        :param pixelscale: pixel scale
        :return:
        """
        # make instance of lenstronomy data class
        x_grid, y_grid, ra_at_xy_0, dec_at_xy_0, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix = util.make_grid_with_coordtransform(numPix=numpix, deltapix=pixelscale, subgrid_res=1,
                                                            left_lower=False, inverse=False)
        kwargs_data = {'numPix': numpix, 'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0,
                       'transform_pix2angle': Mpix2coord}
        data_class = Data(kwargs_data)

        # make instance of lenstronomy PSF class
        if psf_image is None:
            psf_type = "GAUSSIAN"
            fwhm = 0.9
            kwargs_psf = {'psf_type': psf_type, 'fwhm': fwhm}
        else:
            kwargs_psf = {'psf_type': "PIXEL", 'kernel_point_source': psf_image}
        psf_class = PSF(kwargs_psf)
        lens_model_class = LensModel(lens_model_list=['SIS'])
        source_model_class = LightModel(light_model_list=['SERSIC'])
        lens_light_model_class = LightModel(light_model_list=['SERSIC'])
        imageModel = ImageModel(data_class=data_class, psf_class=psf_class, lens_model_class=lens_model_class,
                                source_model_class=source_model_class, lens_light_model_class=lens_light_model_class)
        model = imageModel.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source, kwargs_lens_light=kwargs_lens_light)
        return model
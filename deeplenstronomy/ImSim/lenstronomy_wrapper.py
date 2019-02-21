from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.Data.psf import PSF
from lenstronomy.Data.imaging_data import Data
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.param_util as param_util
import lenstronomy.Util.image_util as image_util
import lenstronomy.Util.util as util
from lenstronomy.LightModel.Profiles.sersic import Sersic

import numpy as np


class LenstronomyAPI(object):
    """
    This class takes as an input physical lensing properties (e.g. by LensPop), in addition to an image data
    configuration from the data module and turns it into an image
    """
    def __init__(self, skySurvey, cosmo=None):
        """

        :param cosmo: astropy.cosmology class
        :param skySurvey: instance of data containing all its information relevant for a given sky survey
        """
        if cosmo is None:
            from astropy.cosmology import default_cosmology
            cosmo = default_cosmology.get()
        self._cosmo = cosmo
        self._pixelsize = skySurvey.pixelsize
        self._magnitude_zero_point = skySurvey.magnitude_zero_point
        self._psf_type = skySurvey.psf_type
        self._psf_fwhm = skySurvey.seeing
        self._psf_model = skySurvey.psf_model
        self._sigma_bkg = skySurvey.sigma_bkg
        self._exposure_time = skySurvey.exposure_time

    def sim_image(self, numpix, z_lens, z_source, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None,
                  kwargs_ps=None, with_noise=True):
        imageModel, kwargs_lens_, kwargs_source_, kwargs_lens_light_, kwargs_ps_ = self.lensPop2lenstronomy(numpix,
                        z_lens, z_source, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
        model = imageModel.image(kwargs_lens=kwargs_lens_, kwargs_source=kwargs_source_,
                                 kwargs_lens_light=kwargs_lens_light_, kwargs_ps=kwargs_ps_)
        if with_noise is True:
            model = self.add_noise(model, self._sigma_bkg, self._exposure_time)
        return model

    def decompose_image(self, numpix, z_lens, z_source, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None,
                  kwargs_ps=None):
        """

        :param numpix:
        :param z_lens:
        :param z_source:
        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :return:
        """
        imageModel, kwargs_lens_, kwargs_source_, kwargs_lens_light_, kwargs_ps_ = self.lensPop2lenstronomy(numpix,
                        z_lens, z_source, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
        model_full = imageModel.image(kwargs_lens=kwargs_lens_, kwargs_source=kwargs_source_,
                                 kwargs_lens_light=kwargs_lens_light_, kwargs_ps=kwargs_ps_)
        model_source = imageModel.image(kwargs_lens=kwargs_lens_, kwargs_source=kwargs_source_,
                                      kwargs_lens_light=kwargs_lens_light_, kwargs_ps=kwargs_ps_, source_add=True,
                                        lens_light_add=False, point_source_add=False)
        model_lens_light = imageModel.image(kwargs_lens=kwargs_lens_, kwargs_source=kwargs_source_,
                                        kwargs_lens_light=kwargs_lens_light_, kwargs_ps=kwargs_ps_, source_add=False,
                                        lens_light_add=True, point_source_add=False)
        model_point_source = imageModel.image(kwargs_lens=kwargs_lens_, kwargs_source=kwargs_source_,
                                        kwargs_lens_light=kwargs_lens_light_, kwargs_ps=kwargs_ps_, source_add=False,
                                        lens_light_add=False, point_source_add=True)
        return model_full, model_source, model_lens_light, model_point_source

    def lensPop2lenstronomy(self, numpix, z_lens, z_source, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None,
                  kwargs_ps=None):
        """
        translates lensPop quantities into lenstronomy quantities given the specific survey
        returns an instance of the ImageModel module of lenstronomy with all the keyword arguments attached

        :param numpix:
        :param z_lens:
        :param z_source:
        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :return:
        """
        if kwargs_lens is None:
            lensModel, kwargs_lens_ = None, None
        else:
            lensModel, kwargs_lens_ = self._lensPop2lenstronomy_lens(z_lens, z_source, **kwargs_lens)
        if kwargs_lens_light is None:
            lensLightModel, kwargs_lens_light_ = None, None
        else:
            lensLightModel, kwargs_lens_light_ = self._lensPop2lenstronomy_sersic(z_lens, pixelsize=self._pixelsize,
                                                                                  **kwargs_lens_light)
        if kwargs_source is None:
            sourceLightModel, kwargs_source_ = None, None
        else:
            sourceLightModel, kwargs_source_ = self._lensPop2lenstronomy_sersic(z_source, pixelsize=self._pixelsize,
                                                                                **kwargs_source)
        if kwargs_ps is None:
            pointSourceModel, kwargs_ps_ = None, None
        else:
            pointSourceModel, kwargs_ps_ = self._lensPop2lenstronomy_point_source(z_source, **kwargs_ps)
        data_class, psf_class = self.data_configure(numpix=numpix)
        imageModel = ImageModel(data_class=data_class, psf_class=psf_class, lens_model_class=lensModel,
                                source_model_class=sourceLightModel, lens_light_model_class=lensLightModel,
                                point_source_class=pointSourceModel)
        return imageModel, kwargs_lens_, kwargs_source_, kwargs_lens_light_, kwargs_ps_

    def data_configure(self, numpix):
        """
        configures the Data() and PSF() class instances of lenstronomy based on the SkySurvey instance

        :param numpix: number of pixel at each side
        :return: data_class, psf_class
        """
        x_grid, y_grid, ra_at_xy_0, dec_at_xy_0, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix = util.make_grid_with_coordtransform(
            numPix=numpix, deltapix=self._pixelsize, subgrid_res=1, left_lower=False, inverse=False)
        kwargs_data = {'numPix': numpix, 'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0,
                       'transform_pix2angle': Mpix2coord}
        data_class = Data(kwargs_data)

        # make instance of lenstronomy PSF class
        if self._psf_type == 'GAUSSIAN':
            psf_type = "GAUSSIAN"
            fwhm = 0.9
            kwargs_psf = {'psf_type': psf_type, 'fwhm': fwhm}
        else:
            kwargs_psf = {'psf_type': "PIXEL", 'kernel_point_source': self._psf_model}
        psf_class = PSF(kwargs_psf)
        return data_class, psf_class

    def add_noise(self, model, sigma_bkg, exposure_time):
        """

        :param sigma_bkg:
        :param exposure_time:
        :return:
        """
        poisson = image_util.add_poisson(model, exp_time=exposure_time)
        bkg = image_util.add_background(model, sigma_bkd=sigma_bkg)
        return model + bkg + poisson

    def _lensPop2lenstronomy_lens(self, z_lens, z_source, velocity_dispersion, axis_ratio=1, inclination_angle=0,
                                  center_ra=0, center_dec=0):
        """
        inputs lensPop lens quantities and returns a lens model instance of lenstronomy and the parameters associated with it



        :param z_lens:
        :param z_source:
        :param velocity_dispersion:
        :param axis_ratio:
        :param inclination_angle:
        :param center_ra:
        :param center_dec:
        :return: lenstronomy lensModel() class, lenstronomy parameters
        """

        lensCosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=self._cosmo)
        vel_dist = velocity_dispersion
        theta_E = lensCosmo.sis_sigma_v2theta_E(vel_dist)

        e1, e2 = param_util.phi_q2_ellipticity(inclination_angle, axis_ratio)

        lensModel = LensModel(lens_model_list=['SIE'])
        kwargs = [{'theta_E': theta_E, 'e1': e1, 'e2': e2, 'center_x': center_ra, 'center_y': center_dec}]
        return lensModel, kwargs

    def _lensPop2lenstronomy_light(self, z_object, kwargs_light):
        """

        :param z_object: redshift of object
        :param kwargs_light: kwargs from lensPop
        :return: light model instance and lenstronomy kwargs list
        """
        if 'image' in kwargs_light:
            lightModel, kwargs_light_ = self._configure_image(**kwargs_light)
        elif 'n_sersic' in kwargs_light:
            lightModel, kwargs_light_ = self._lensPop2lenstronomy_sersic(z_object, **kwargs_light)
        else:
            raise ValueError("keyword arguments %s can not be matched to lenstronomy light model" % kwargs_light)
        return lightModel, kwargs_light_

    def _lensPop2lenstronomy_sersic(self, z_object, magnitude, halflight_radius, n_sersic, axis_ratio=1,
                                    inclination_angle=0, center_ra=0, center_dec=0, pixelsize=1,
                                    apparent_magnitude=True, arcsecond_scales=True):
        """
        computes lenstronomy conventions for the light profile

        :param z_object: redshift of object
        :param magnitude: magnitude of object
        :param halflight_radius: half light radius in physical kpc
        :param magnitude_zero_point: magnitude zero point of the observations
        :param pixelsize: pixel size of data
        :return: half light radius in angles, flux amplitude at half light radius
        """

        # convert absolute to apparent magnitude
        if apparent_magnitude is False:
            apparent_magnitude = self._abs2apparent_magnitude(magnitude, z_object)
        else:
            apparent_magnitude = magnitude
        # convert magnitude in counts per second
        cps = self._mag2cps(apparent_magnitude, self._magnitude_zero_point)

        # convert physical half light radius into angle
        if arcsecond_scales is False:
            lensCosmo = LensCosmo(z_source=z_object, z_lens=z_object)
            Rh_angle = lensCosmo.phys2arcsec_lens(phys=halflight_radius/1000.)
        else:
            Rh_angle = halflight_radius

        # convert total counts per second into counts per second of a pixel at the half light radius
        amp = self._cps2lenstronomy_amp(cps, Rh_angle, n_sersic, pixelsize)
        e1, e2 = param_util.phi_q2_ellipticity(inclination_angle, axis_ratio)
        lightModel = LightModel(light_model_list=['SERSIC_ELLIPSE'])
        kwargs = [{'amp': amp, 'R_sersic': Rh_angle, 'n_sersic': n_sersic, 'e1': e1, 'e2': e2,
                   'center_x': center_ra, 'center_y': center_dec}]
        return lightModel, kwargs

    def _lensPop2lenstronomy_point_source(self, z_object, magnitude, center_ra, center_dec, apparent_magnitude=True):
        """

        :param z_source: redshift of point source
        :param magnitude: magnitude of intrinsic source
        :param center_ra: position in angle of intrinsic source
        :param center_dec: position in angle of intrinsic source
        :return:
        """
        if apparent_magnitude is False:
            apparent_magnitude = self._abs2apparent_magnitude(magnitude, z_object)
        else:
            apparent_magnitude = magnitude
        cps = self._mag2cps(apparent_magnitude, self._magnitude_zero_point)
        pointSource = PointSource(point_source_type_list=['SOURCE_POSITION'])
        kwargs = [{'source_amp': cps, 'ra_source': center_ra, 'dec_source': center_dec}]
        return pointSource, kwargs

    def _abs2apparent_magnitude(self, absolute_magnitude, z_object):
        """
        converts absolute to apparent magnitudes

        :param absolute_magnitude: absolute magnitude of object
        :param z_object: redshift of object
        :return: apparent magnitude
        """
        # physical distance in Mpc
        D_L_Mpc = self._cosmo.luminosity_distance(z_object)
        D_L = D_L_Mpc * 10**6  # physical distance in parsec
        m_apparent = 5.8 * (np.log10(D_L) - 1) + absolute_magnitude
        return m_apparent

    def _mag2cps(self, magnitude, magnitude_zero_point):
        """
        converts an apparent magnitude to counts per second

        The zero point of an instrument, by definition, is the magnitude of an object that produces one count
        (or data number, DN) per second. The magnitude of an arbitrary object producing DN counts in an observation of
        length EXPTIME is therefore:
        m = -2.5 x log10(DN / EXPTIME) + ZEROPOINT

        :param magnitude:
        :param magnitude_zero_point:
        :return: counts per second
        """
        delta_M = magnitude - magnitude_zero_point
        counts = 10**(-delta_M/2.5)
        return counts

    def _cps2lenstronomy_amp(self, cps, Rh_angle, n_sersic, pixelsize):
        """

        :param cps: total counts per second of object
        :param Rh_angle: half light radius (in arcseconds) of object
        :param pixelsize: pixel size of data
        :return: cps value of pixel at the half light radius
        """
        # compute norm flux of Sersic profile
        sersic_util = Sersic()
        flux_norm = sersic_util.total_flux(r_eff=Rh_angle, I_eff=1, n_sersic=n_sersic)
        # convert to number measured (with I_eff in arcseconds)
        I_eff_measured = cps / flux_norm
        # convert in surface brightness per pixel
        amp = I_eff_measured
        return amp

    def _configure_image(self, image, pixelsize, magnitude, relative_rotation, center_ra, center_dec):
        """

        :param image: a pixelized image
        :param pixelsize: pixel size (in angular units) to be used in the simulated image
        :param magnitude: apparent magnitude to simulated image
        :param relative_rotation: relative rotation angle (radian)
        :param center_ra: center of simulated image
        :param center_dec: center of simulated image
        :return: lightModel instance and keyword arguments of the model
        """
        lightModel = LightModel(light_model_list=['INTERPOL'])
        cps = self._mag2cps(magnitude, self._magnitude_zero_point)
        image_normed = image / np.sum(image)
        kwargs = [{'image': image_normed, 'amp': cps, 'center_x': center_ra, 'center_y': center_dec,
                   'phi_G': relative_rotation, 'scale': pixelsize}]
        return lightModel, kwargs

# A module to simulate a desired number of images based on quasar light curves

import numpy as np
from astropy.cosmology import FlatLambdaCDM
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.param_util as param_util
import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.image_util as image_util
from lenstronomy.Util import kernel_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
import lenstronomy.Util.data_util as data_util
import lenstronomy.Util.util as util
import lenstronomy.Plots.plot_util as plot_util
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.LightModel.Profiles.gaussian import GaussianEllipse

def mag_to_flux(m, mz):
    return 10**((mz - m)/2.5)

"""
Function to simulate lensed quasars. Will produce doubles and quads. Quads may look like triples in some cases.

Input: a row in the merged_lcs csv files.

Output: griz images based on input 
"""

def sim_lens(data, numPix=101, sigma_bkg=8.0, exp_time=100.0, deltaPix=0.263, psf_type='GAUSSIAN', kernel_size = 91):
    
    flux_g = mag_to_flux(data['mag_g'],27.5)
    flux_r = mag_to_flux(data['mag_r'],27.5)
    flux_i = mag_to_flux(data['mag_i'],27.5)
    flux_z = mag_to_flux(data['mag_z'],27.5)
    flux_source = mag_to_flux(data['source_mag'], 27.5)
    flux_lens = mag_to_flux(data['lens_mag'], 27.5)
    
    color_idx = {'g': 0, 'r': 1, 'i': 2, 'z': 3}
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.)
    
    full_band_images = np.zeros((numPix, numPix, 4))
    
    ### Set kwargs based on input data file
    #shear
    kwargs_shear = {'gamma_ext': data['lens_shear_gamma_ext'], 'psi_ext': data['lens_shear_psi_ext']} 
    #lens potential
    kwargs_spemd = {'theta_E': data['lens_theta_E'], 'gamma': data['lens_gamma'], 'center_x': data['lens_center_x'], 
                    'center_y': data['lens_center_y'], 'e1': data['lens_e1'], 'e2': data['lens_e2']}
    #lens light
    kwargs_sersic_lens = {'amp': flux_lens, 'R_sersic': data['lens_R_sersic'], 'n_sersic': data['lens_n_sersic'], 
                          'e1': data['lens_e1'], 'e2': data['lens_e2'], 'center_x': data['lens_center_x'], 
                          'center_y': data['lens_center_y']}
    #source
    kwargs_sersic_source = {'amp': flux_source, 'R_sersic': data['source_R_sersic'], 'n_sersic': data['source_n_sersic'], 
                            'e1': data['source_e1'], 'e2': data['source_e2'], 'center_x': data['source_center_x'], 
                            'center_y': data['source_center_y']}
    
    ###set model parameters based on kwargs
    #lens potential
    lens_model_list = ['SPEP', 'SHEAR_GAMMA_PSI']
    kwargs_lens = [kwargs_spemd, kwargs_shear]
    lens_model_class = LensModel(lens_model_list=lens_model_list)
    #lens light
    lens_light_model_list = ['SERSIC_ELLIPSE']
    kwargs_lens_light = [kwargs_sersic_lens]
    lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
    #source
    source_model_list = ['SERSIC_ELLIPSE']
    kwargs_source = [kwargs_sersic_source]
    source_model_class = LightModel(light_model_list=source_model_list)
    
    ###configure image based on data properties
    kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, sigma_bkg)
    data_class = ImageData(**kwargs_data)
    
    ###solve lens equation
    lensEquationSolver = LensEquationSolver(lens_model_class)
    x_image, y_image = lensEquationSolver.findBrightImage(kwargs_sersic_source['center_x'], 
                                                          kwargs_sersic_source['center_y'], 
                                                          kwargs_lens, numImages=4, min_distance=deltaPix, 
                                                          search_window=numPix * deltaPix)
    magnification = lens_model_class.magnification(x_image, y_image, kwargs=kwargs_lens)
    
    ###iterate through bands to simulate images
    for band in ['g', 'r', 'i', 'z']:
         
        #psf info
        kwargs_psf = {'psf_type': psf_type, 'fwhm':data['psf_%s'%band], 'pixel_size': deltaPix, 'truncation': 3}
        psf_class = PSF(**kwargs_psf)
        
        #quasar info
        kwargs_ps = [{'ra_image': x_image, 'dec_image': y_image,
                      'point_amp': np.abs(magnification)*eval('flux_%s'%band)}]
        point_source_list = ['LENSED_POSITION']
        point_source_class = PointSource(point_source_type_list=point_source_list, fixed_magnification_list=[False])
        
        #build image model
        kwargs_numerics = {'supersampling_factor': 1}
        imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class,
                                lens_light_model_class, point_source_class, kwargs_numerics=kwargs_numerics)
        
        #generate image
        image_sim = imageModel.image(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
        poisson = image_util.add_poisson(image_sim, exp_time=exp_time)
        bkg = image_util.add_background(image_sim, sigma_bkd=sigma_bkg)
        image_sim = image_sim + bkg + poisson
        
        data_class.update_data(image_sim)
        kwargs_data['image_data'] = image_sim


        kwargs_model = {'lens_model_list': lens_model_list,
                        'lens_light_model_list': lens_light_model_list,
                        'source_light_model_list': source_model_list,
                        'point_source_model_list': point_source_list
                         }
        #build up an array with one slice for each band
        full_band_images[:, :, color_idx[band]] += image_sim
        
    return full_band_images

"""
Function to average images generated from multiple rows of the same AGN light curve.
This function calls the lensing image simulation function.

Input: dataframe of merged_lcs file information

Output: mean griz images of all observations in light curve
"""

def generate_mean_sim_lens(data):
    
    images = []
    for index, row in data.iterrows():
        images.append(sim_lens(row))
        
    im_arr = np.array(images)
    mean_images = np.mean(im_arr, axis=0)
    
    return mean_images


"""
Function to smimulate foreground, non-lensed agn based on light curves.

input: agn light curve row and positions of the foreground galaxies

output: griz images
"""

def sim_non_lens_agn(data, center_x_1, center_y_1, center_x_2, center_y_2, numPix=101, sigma_bkg=8.0, exp_time=100.0, deltaPix=0.263, psf_type='GAUSSIAN', kernel_size = 91):
    
    full_band_images = np.zeros((numPix, numPix, 4))
    
    flux_1 = mag_to_flux(data['source_mag'], 27.5)
    flux_g_1 = mag_to_flux(data['mag_g'],27.5)
    flux_r_1 = mag_to_flux(data['mag_r'],27.5)
    flux_i_1 = mag_to_flux(data['mag_i'],27.5)
    flux_z_1 = mag_to_flux(data['mag_z'],27.5)
    
    flux_2 = mag_to_flux(data['lens_mag'], 27.5)
    flux_g_2 = flux_g_1 * flux_2 / flux_1
    flux_r_2 = flux_r_1 * flux_2 / flux_1
    flux_i_2 = flux_i_1 * flux_2 / flux_1
    flux_z_2 = flux_z_1 * flux_2 / flux_1
    
    center_x_list = [center_x_1, center_x_2]
    center_y_list = [center_y_1, center_y_2]
    
    kwargs_gal1 = {'amp': flux_1, 'R_sersic': data['source_R_sersic'], 'n_sersic': data['source_n_sersic'],
                   'e1': data['source_e1'], 'e2': data['source_e2'], 'center_x': center_x_1, 'center_y': center_y_1}
    kwargs_gal2 = {'amp': flux_2, 'R_sersic': data['lens_R_sersic'], 'n_sersic': data['lens_n_sersic'],
                   'center_x': center_x_2, 'center_y': center_y_2}
    kwargs_gals = [kwargs_gal1, kwargs_gal2]
    
    color_idx = {'g': 0, 'r': 1, 'i': 2, 'z': 3}
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.)
    
    light_model_list = ['SERSIC_ELLIPSE', 'SERSIC']
    lightModel = LightModel(light_model_list=light_model_list)
    
    ###iterate through bands to simulate images
    for band in ['g', 'r', 'i', 'z']:
         
        #psf info
        kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, sigma_bkg)
        data_class = ImageData(**kwargs_data)
        kwargs_psf = {'psf_type': psf_type, 'fwhm':data['psf_%s'%band], 'pixel_size': deltaPix, 'truncation': 3}
        psf_class = PSF(**kwargs_psf)
    
        point_source_list = ['UNLENSED']
        pointSource = PointSource(point_source_type_list=point_source_list)
        
        point_amp_list = [eval('flux_%s_1'%band), eval('flux_%s_2'%band)]
        kwargs_ps = [{'ra_image': center_x_list, 'dec_image': center_y_list, 'point_amp': point_amp_list}]
        kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}
        imageModel = ImageModel(data_class, psf_class, lens_light_model_class=lightModel,
                                    point_source_class=pointSource, kwargs_numerics=kwargs_numerics)

    
        # generate image
        image_sim = imageModel.image(kwargs_lens_light=kwargs_gals, kwargs_ps=kwargs_ps)
        poisson = image_util.add_poisson(image_sim, exp_time=exp_time)
        bkg = image_util.add_background(image_sim, sigma_bkd=sigma_bkg)
        image_sim = image_sim + bkg + poisson
        
        full_band_images[:, :, color_idx[band]] += image_sim
        
    return full_band_images


"""
Function to generate mean non-lensed AGN. Function calls sim_non_lens-agn

input: dataframe of merged light curve

output: mean griz images
"""

def generate_mean_sim_non_lens_agn(data):
    
    center_x_1 = np.random.uniform(-0.03, 0.03)
    center_y_1 = np.random.uniform(-0.03, 0.03)
    center_x_2 = np.random.uniform(-4.3, 4.3)
    center_y_2 = np.random.uniform(-4.3, 4.3)
    
    images = []
    for index, row in data.iterrows():
        images.append(sim_non_lens_agn(row, center_x_1, center_y_1, center_x_2, center_y_2))
        
    im_arr = np.array(images)
    mean_images = np.mean(im_arr, axis=0)
    
    return mean_images

"""
Function to simulate single galaxies (not AGN)

input: row in merged lcs for flux and geometric information

output: griz images of galaxy near center

"""

def sim_non_lens_gal(data,  numPix=101, sigma_bkg=8.0, exp_time=100.0, deltaPix=0.263, psf_type='GAUSSIAN', kernel_size = 91):

    full_band_images = np.zeros((numPix, numPix, 4))
    center_x_1 = np.random.uniform(-0.03, 0.03)
    center_y_1 = np.random.uniform(-0.03, 0.03)
    flux_1 = mag_to_flux(data['source_mag'], 27.5)
    flux_2 = mag_to_flux(data['lens_mag'], 27.5)
    
    light_model_list = ['SERSIC_ELLIPSE', 'SERSIC']
    lightModel = LightModel(light_model_list=light_model_list)
    
    kwargs_disk = {'amp': flux_1, 'R_sersic': data['source_R_sersic'], 'n_sersic': data['source_n_sersic'],
                   'e1': data['source_e1'], 'e2': data['source_e2'], 'center_x': center_x_1, 'center_y': center_y_1}
    kwargs_bulge = {'amp': flux_2, 'R_sersic': data['lens_R_sersic'], 'n_sersic': data['lens_n_sersic'],
                   'center_x': center_x_1, 'center_y': center_y_1}
    
    kwargs_host = [kwargs_disk, kwargs_bulge]
    
    color_idx = {'g': 0, 'r': 1, 'i': 2, 'z': 3}
    
    for band in ['g', 'r', 'i', 'z']:
        
        kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, sigma_bkg)
        data_class = ImageData(**kwargs_data)
        kwargs_psf = {'psf_type': psf_type, 'fwhm':data['psf_%s'%band], 'pixel_size': deltaPix, 'truncation': 3}
        psf_class = PSF(**kwargs_psf)
        
        kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}
        imageModel = ImageModel(data_class, psf_class, lens_light_model_class=lightModel,
                                     kwargs_numerics=kwargs_numerics)
        
        image_sim = imageModel.image(kwargs_lens_light=kwargs_host)
        poisson = image_util.add_poisson(image_sim, exp_time=exp_time)
        bkg = image_util.add_background(image_sim, sigma_bkd=sigma_bkg)
        image_sim = image_sim + bkg + poisson
        
        full_band_images[:, :, color_idx[band]] += image_sim

    return full_band_images
    

"""
Function to average non-lensed galaxies. Function calls sim_non_lens_gal

input: light curve dataframe

output: mean griz images
"""

def generate_mean_sim_non_lens_gal(data):
    
    images = []
    for index, row in data.iterrows():
        images.append(sim_non_lens_gal(row))
        
    im_arr = np.array(images)
    mean_images = np.mean(im_arr, axis=0)
    
    return mean_images

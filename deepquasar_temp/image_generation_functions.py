# A module to simulate and average images based pn parameter_playground.ipynb

#### Warning: very hacky code

import pandas as pd
import sys
import utils

import copy
import numpy as np
import aplpy
from PIL import Image
import matplotlib.pyplot as plt

import lenstronomy.Util.data_util as data_util
import lenstronomy.Util.util as util
import lenstronomy.Plots.plot_util as plot_util
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.LightModel.Profiles.gaussian import GaussianEllipse
gauss = GaussianEllipse()

### Random parameter choice functions
def get_n_sersic():
    return np.random.uniform(low=0.5, high=10.0)

def get_R_sersic():
    return np.random.uniform(low=0.5, high=5.0)

def get_e12():
    return np.random.uniform(low=-0.4, high=0.4)

def get_center_xy():
    return np.random.uniform(low=-1.0, high=1.0)

def get_ra_dec_dev():
    temp = np.random.uniform(low=-0.5, high=0.5)
    if temp > 0.0:
        return np.random.uniform(low=0.2, high=0.2)
    else:
        return np.random.uniform(low=-0.5, high=-0.5)

def get_gal_ab_mag():
    return np.random.uniform(low=16, high=18)

def pick_z_lens(z_source):
    z_lens = np.random.uniform(low=0.05, high=z_source - 0.01)
    return z_lens

def get_theta_E(z_source, z_lens, lens_mass=2.3e22):
    assert z_source > z_lens
    
    DL = utils.convert_z_to_d(np.array([z_source, z_lens]))
    DLs = DL[0]
    DLl = DL[1]
    
    DAs = DLs / (1 + z_source)**2
    DAl = DLl / (1 + z_lens)**2
    
    distance_ratio = (DAs - DAl) / (DAs * DAl) #units of pc-1
    distance_ratio_meters = distance_ratio / 3.086e16 #units of m-1
    
    G = 6.67408e-11 #units m3 kg-1 s-2
    c = 299792458.0 #units m s-1
    Msolar = 1.989e30 #units kg
    M = lens_mass * Msolar
    
    theta_E_squared = 4 * G * M * distance_ratio_meters / c**2
    
    return np.sqrt(theta_E_squared)
    return #np.log(2 * (theta_E + 1)) ##hack until theta_E gets figured out

def sim_lens_geometry(z_source, z_lens):
    
    center_x, center_y = get_center_xy(), get_center_xy()
    e1, e2 = get_e12(), get_e12()
    
    lens_kwargs = {'magnitude': get_gal_ab_mag(), 
                   'R_sersic': get_R_sersic(),
                   'n_sersic': get_n_sersic(),
                   'e1': e1,
                   'e2': e2,
                   'center_x': center_x,
                   'center_y': center_y}
    
    theta_E = get_theta_E(z_source, z_lens)
    
    lens_model_kwargs_sie = {'theta_E': theta_E,
                             'e1': e1,
                             'e2': e2,
                             'center_x': center_x,
                             'center_y': center_y}
    
    lens_model_kwargs_shear = {'e1': 0.03, 'e2': 0.01} 
    
    return lens_kwargs, lens_model_kwargs_sie, lens_model_kwargs_shear

def sim_source_geometry():
    source_kwargs = {'magnitude': get_gal_ab_mag() - 1,
                     'R_sersic': get_R_sersic(),
                     'n_sersic': get_n_sersic(),
                     'e1': get_e12(),
                     'e2': get_e12(),
                     'center_x': get_center_xy(),
                     'center_y': get_center_xy()}
    
    return source_kwargs

def sim_point_source(center_x, center_y):
    ps_kwargs = {'ra_source': center_x + 1.1 * get_ra_dec_dev(), 
                 'dec_source': center_y + 1.1 * get_ra_dec_dev()}
    return ps_kwargs
    
    
def write_geo(outfile, geo_dict):
    np.save(outfile + '_geo.npy', geo_dict)
    return
    

def sim_geometry(z_source, outfile=None):
    #pick lens z
    z_lens = pick_z_lens(z_source)
    
    #lens
    lens_kwargs, lens_model_kwargs_sie, lens_model_kwargs_shear = sim_lens_geometry(z_source, z_lens)
    
    #source
    source_kwargs = sim_source_geometry()
    
    #point source
    ps_kwargs = sim_point_source(source_kwargs['center_x'], source_kwargs['center_y'])
    
    #make dictionary
    geo_dict = {'z_source': z_source,
                'z_lens': z_lens,
                'lens_kwargs': lens_kwargs,
                'lens_model_kwargs_sie': lens_model_kwargs_sie,
                'lens_model_kwargs_shear': lens_model_kwargs_shear,
                'source_kwargs': source_kwargs,
                'ps_kwargs': ps_kwargs}
    
    #write geometry to a file
    if outfile is not None:
        write_geo(outfile, geo_dict)
    
    return geo_dict

### Forced geometry parameter overwrite functions
def overwrite_params(dictionary, param, new_value):
    dictionary[param] = new_value
    return dictionary

def force_larger_ps_separation(geo_dict):
    geo_dict['ps_kwargs'] = overwrite_params(geo_dict['ps_kwargs'], 
                                             'ra_source',
                                             (geo_dict['ps_kwargs']['ra_source'] - geo_dict['source_kwargs']['center_x']) * 1.4 + geo_dict['source_kwargs']['center_x'])
    geo_dict['ps_kwargs'] = overwrite_params(geo_dict['ps_kwargs'], 
                                             'dec_source',
                                             (geo_dict['ps_kwargs']['dec_source'] - geo_dict['source_kwargs']['center_y']) * 1.4 + geo_dict['source_kwargs']['center_y'])
    return geo_dict

#Label 2: One galaxy with SN
def force_no_lensing_with_ps(geo_dict):
    #must be used inside sim_single_images so geo_dict has mag info for point source
    #fake no lensing by moving ps to lens and turning off source
    geo_dict['ps_kwargs'] = overwrite_params(geo_dict['ps_kwargs'], 
                                             'ra_source', 
                                             geo_dict['lens_kwargs']['center_x'] + 3 * geo_dict['ps_kwargs']['ra_source'] - geo_dict['source_kwargs']['center_x'])
    geo_dict['ps_kwargs'] = overwrite_params(geo_dict['ps_kwargs'], 
                                             'dec_source', 
                                             geo_dict['lens_kwargs']['center_y'] + 3 * geo_dict['ps_kwargs']['dec_source'] - geo_dict['source_kwargs']['center_y'])
    geo_dict['lens_model_kwargs_sie']['theta_E'] = 1e-6
    geo_dict['source_kwargs'] = overwrite_params(geo_dict['source_kwargs'], 'magnitude', 90)
    return geo_dict

#Label 3: Two galaxies, no SN
def force_lensed_agn(geo_dict):
    #must be used inside sim_single_images so geo_dict has mag info for point source
    #fake lensed agn by moving ps in source to center of source
    geo_dict['ps_kwargs'] = overwrite_params(geo_dict['ps_kwargs'], 
                                             'ra_source', 
                                             geo_dict['source_kwargs']['center_x'])
    geo_dict['ps_kwargs'] = overwrite_params(geo_dict['ps_kwargs'], 
                                             'dec_source', 
                                             geo_dict['source_kwargs']['center_y'])
    geo_dict['ps_kwargs'] = overwrite_params(geo_dict['ps_kwargs'], 'magnitude', geo_dict['ps_kwargs']['magnitude'] + 2)
    return geo_dict

#Label 4: One galaxy, no SN
def force_non_lensed_agn(geo_dict):
    #move ps to lens center and turn off source
    geo_dict['ps_kwargs'] = overwrite_params(geo_dict['ps_kwargs'], 
                                             'ra_source', 
                                             geo_dict['lens_kwargs']['center_x'])
    geo_dict['ps_kwargs'] = overwrite_params(geo_dict['ps_kwargs'], 
                                             'dec_source', 
                                             geo_dict['lens_kwargs']['center_y'])
    geo_dict['ps_kwargs'] = overwrite_params(geo_dict['ps_kwargs'], 'magnitude', geo_dict['ps_kwargs']['magnitude'] + 2)
    geo_dict['lens_model_kwargs_sie']['theta_E'] = 1e-6
    geo_dict['source_kwargs'] = overwrite_params(geo_dict['source_kwargs'], 'magnitude', 90)
    return geo_dict


### Image siumualtion
def sim_single_image(sn_data, geo_dict, label):
    
    DECam = {'read_noise': 10,      # std of noise generated by read-out (in units of electrons)                                                   
             'pixel_scale': 0.263,  # scale (in arcseonds) of pixels                                                                               
             'ccd_gain': 4.5        # electrons/ADU (analog-to-digital unit).                                                                      
              }

    obs_info = {'exposure_time': 90,
                'sky_brightness': sn_data['SKYMAG'],
                'magnitude_zero_point': sn_data['ZPTMAG'],
                'num_exposures': 1,
                'seeing': sn_data['PSF'] / 3,
                'psf_type': 'GAUSSIAN'}

    kwargs_model = {'lens_model_list': ['SIE', 'SHEAR'],  # list of lens models to be used                                                         
                'lens_light_model_list': ['SERSIC_ELLIPSE'],  # list of unlensed light models to be used                                           
                'source_light_model_list': ['SERSIC_ELLIPSE'],  # list of extended source models to be used                                        
                'point_source_model_list': ['SOURCE_POSITION'],  # list of point source models to be used                                          
                }

    numpix = 64
    kwargs_merged = util.merge_dicts(DECam, obs_info)
    kwargs_numerics = {'point_source_supersampling_factor': 1}

    sim = SimAPI(numpix=numpix,
                 kwargs_single_band=kwargs_merged,
                 kwargs_model=kwargs_model,
                 kwargs_numerics=kwargs_numerics)

    imSim = sim.image_model_class

    x_grid, y_grid = util.make_grid(numPix=10, deltapix=1)
    flux = gauss.function(x_grid, y_grid, amp=1, sigma=2, e1=0.4, e2=0, center_x=0, center_y=0)
    image_gauss = util.array2image(flux)
    
    ## Set point source mag
    geo_dict['ps_kwargs']['magnitude'] = sn_data['ABMAG'] * 1.4
    
    if label == 1:
        geo_dict = force_larger_ps_separation(geo_dict)
    elif label == 2:
        geo_dict = force_no_lensing_with_ps(geo_dict)
    elif label == 3:
        geo_dict = force_lensed_agn(geo_dict)
    elif label == 4:
        geo_dict = force_non_lensed_agn(geo_dict)
    else:
        print("Unexpected option passed as class label")
        sys.exit()

    #lens light 
    kwargs_lens_light_mag = [geo_dict['lens_kwargs']]
    # source light  
    kwargs_source_mag = [geo_dict['source_kwargs']]
    # point source  
    kwargs_ps_mag = [geo_dict['ps_kwargs']]


    kwargs_lens_light, kwargs_source, kwargs_ps = sim.magnitude2amplitude(kwargs_lens_light_mag, kwargs_source_mag, kwargs_ps_mag)
    kwargs_lens = [geo_dict['lens_model_kwargs_sie'], geo_dict['lens_model_kwargs_shear']]

    image = imSim.image(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
    image += sim.noise_for_model(model=image)

    return np.log10(image + 2)
    #return image

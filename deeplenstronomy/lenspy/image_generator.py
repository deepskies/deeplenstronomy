# Generate images from the organized user inputs

from astropy.cosmology import FlatLambdaCDM
from lenstronomy.SimulationAPI.sim_api import SimAPI
import numpy as np

import distributions

def dict_select(input_dict, keys):
    """
    Trim a dictionary down to selected keys
    
    :param input_dict: full dictionary
    :param keys: list of keys desired in trimmed dict
    :return: dict: trimmed dictionary
    """
    return {k: input_dict[k] for k in keys}

def dict_select_choose(input_dict, keys):
    """
    Trim a dictionary down to selected keys, if they are in the dictionary
    
    :param input_dict: full dictionary
    :param keys: list of keys desired in trimmed dict
    :return: dict: trimmed dictionary
    """
    return {k: input_dict[k] for k in keys if k in input_dict.keys()}

def select_params(input_dict, profile_prefix):
    """
    Get just the parameters and values for a given profile prefix
    
    :param input_dict: full dictionary
    :param profile_prefix: i.e. "PLANE_1-OBJECT_2-LIGHT_PROFILE_1-"
    :return: dict: parameter dictionary for profile
    """
    params = [k for k in input_dict.keys() if k[0:len(profile_prefix)] == profile_prefix]
    return {x.split('-')[-1]: input_dict[x] for x in params if x[-4:] != 'NAME'}

class ImageGenerator():
    def __init__(self):
        return
    
    def sim_image(self, info_dict):
        """
        Simulate an image based on specifications in sim_dict
        
        :param info_dict: A single element of the output form Organizer.breakup()
        """
        output_image = []
        
        #set the cosmology
        cosmology_info = ['H0', 'Om0', 'Tcmb0', 'Neff', 'm_nu', 'Ob0']
        cosmo = FlatLambdaCDM(**dict_select_choose(list(info_dict.values())[0], cosmology_info))
        
        for band, sim_dict in info_dict.items():
            ### Geometry-independent image properties
            
            # Single Band Info
            single_band_info = ['read_noise', 'pixel_scale', 'ccd_gain', 'exposure_time',
                                'sky_brightness', 'seeing', 'magnitude_zero_point',
                                'num_exposures', 'data_count_unit', 'background_noise']
            kwargs_single_band = dict_select_choose(sim_dict, single_band_info)
            
            # Extinction
            extinction_class = None  #figure this out later
            
            # Numerics -- only use single number kwargs
            numerics_info = ['supersampling_factor', 'compute_mode', 'supersampling_convolution',
                             'supersampling_kernel_size', 'point_source_supersampling_factor', 
                             'convolution_kernel_size']
            kwargs_numerics = dict_select_choose(sim_dict, numerics_info)
            
            ### Geometry-dependent image properties
            
            # Dictionary for physical model
            kwargs_model = {'lens_model_list': [],  # list of lens models to be used
                            'lens_redshift_list': [],  # list of redshift of the deflections
                            'lens_light_model_list': [],  # list of unlensed light models to be used
                            'source_light_model_list': [],  # list of extended source models to be used
                            'source_redshift_list': [],  # list of redshfits of the sources in same order as source_light_model_list
                            'point_source_model_list': [], # list of point source models
                            'cosmo': cosmo,
                            'z_source': 0.0}

            # Lists for model kwargs
            kwargs_source_list, kwargs_lens_model_list, kwargs_lens_light_list, kwargs_point_source_list = [], [], [], []
            
            for plane_num in range(1, sim_dict['NUMBER_OF_PLANES'] + 1):
                
                for obj_num in range(1, sim_dict['PLANE_{0}-NUMBER_OF_OBJECTS'.format(plane_num)] + 1):
                    
                    prefix = 'PLANE_{0}-OBJECT_{1}-'.format(plane_num, obj_num)
                    
                    ### Point sources
                    if sim_dict[prefix + 'HOST'] != 'None':
                        
                        # Foreground objects
                        if sim_dict[prefix + 'HOST'] == 'Foreground':
                            kwargs_model['point_source_model_list'].append('UNLENSED')
                            ps_info = ['PLANE_{0}-OBJECT_{1}-{2}'.format(plane_num, obj_num, x) for x in ['ra_image', 'dec_image', 'magnitude']]
                            kwargs_point_source_list.append(dict_select(sim_dict, ps_info))
                        # Real point sources
                        else:
                            if plane_num < sim_dict['NUMBER_OF_PLANES']:
                                # point sources in the lens
                                kwargs_model['point_source_model_list'].append('LENSED_POSITION')
                                ps_info = ['PLANE_{0}-OBJECT_{1}-{2}'.format(plane_num, obj_num, x) for x in ['ra', 'dec', 'magnitude']]
                                ps_dict_info = dict_select(sim_dict, ps_info)
                                kwargs_point_source_list.append({'ra_image': [ps_dict_info['ra']], 'dec_image': [ps_dict_info['dec']], 'magnitude': [ps_dict_info['magnitude']]})
                            elif plane_num == sim_dict['NUMBER_OF_PLANES']:
                                # point sources in the source plane
                                kwargs_model['point_source_model_list'].append('SOURCE_POSITION')
                                ps_info = ['PLANE_{0}-OBJECT_{1}-{2}'.format(plane_num, obj_num, x) for x in ['ra', 'dec', 'magnitude']]
                                ps_dict_info = dict_select(sim_dict, ps_info) 
                                kwargs_point_source_list.append({'ra_source': ps_dict_info['PLANE_{0}-OBJECT_{1}-ra'.format(plane_num, obj_num)], 
                                                                 'dec_source': ps_dict_info['PLANE_{0}-OBJECT_{1}-dec'.format(plane_num, obj_num)], 
                                                                 'magnitude': ps_dict_info['PLANE_{0}-OBJECT_{1}-magnitude'.format(plane_num, obj_num)]})
                            else:
                                #should never get here
                                assert False
                                
                        # the number of profiles will be zero for point sources, so just skip ahead
                        continue
                    
                    ### Model keywords
                    # All planes except last one - treat as lens
                    if plane_num < sim_dict['NUMBER_OF_PLANES']:
                        for light_profile_num in range(1, sim_dict[prefix + 'NUMBER_OF_LIGHT_PROFILES'] +1):
                            kwargs_model['lens_light_model_list'].append(sim_dict[prefix + 'LIGHT_PROFILE_{0}-NAME'.format(light_profile_num)])
                            #kwargs_model['lens_redshift_list'].append(sim_dict[prefix + 'REDSHIFT'])
                            kwargs_lens_light_list.append(select_params(sim_dict, prefix + 'LIGHT_PROFILE_{0}-'.format(light_profile_num)))
                        for mass_profile_num in range(1, sim_dict[prefix + 'NUMBER_OF_MASS_PROFILES'] +1):
                            kwargs_model['lens_model_list'].append(sim_dict[prefix + 'MASS_PROFILE_{0}-NAME'.format(mass_profile_num)])
                            kwargs_model['lens_redshift_list'].append(sim_dict[prefix + 'REDSHIFT'])
                            kwargs_lens_model_list.append(select_params(sim_dict, prefix + 'MASS_PROFILE_{0}-'.format(mass_profile_num)))
                        for shear_profile_num in range(1, sim_dict[prefix + 'NUMBER_OF_SHEAR_PROFILES'] +1):
                            kwargs_model['lens_model_list'].append(sim_dict[prefix + 'SHEAR_PROFILE_{0}-NAME'.format(shear_profile_num)])
                            kwargs_model['lens_redshift_list'].append(sim_dict[prefix + 'REDSHIFT'])
                            kwargs_lens_model_list.append(select_params(sim_dict, prefix + 'SHEAR_PROFILE_{0}-'.format(shear_profile_num)))
                                                   
                    # Last Plane - treat as source
                    elif plane_num == sim_dict['NUMBER_OF_PLANES']:
                        kwargs_model['z_source'] = sim_dict[prefix + 'REDSHIFT']
                        for light_profile_num in range(1, sim_dict[prefix + 'NUMBER_OF_LIGHT_PROFILES'] +1):
                            kwargs_model['source_light_model_list'].append(sim_dict[prefix + 'LIGHT_PROFILE_{0}-NAME'.format(light_profile_num)])
                            kwargs_model['source_redshift_list'].append(sim_dict[prefix + 'REDSHIFT'])
                            kwargs_source_list.append(select_params(sim_dict, prefix + 'LIGHT_PROFILE_{0}-'.format(light_profile_num)))
                        
                    else:
                        # Should never get here
                        assert False

            # Make image
            sim = SimAPI(numpix=sim_dict['numPix'], 
                         kwargs_single_band=kwargs_single_band, 
                         kwargs_model=kwargs_model)
            imSim = sim.image_model_class(kwargs_numerics)
            
            kwargs_lens_light_list, kwargs_source_list, kwargs_point_source_list = sim.magnitude2amplitude(kwargs_lens_light_mag=kwargs_lens_light_list,
                                                                                                           kwargs_source_mag=kwargs_source_list,
                                                                                                           kwargs_ps_mag=kwargs_point_source_list)
            
            #kwargs_lens_model_list = sim.physical2lensing_conversion(kwargs_mass=kwargs_lens_model_list)
                                                                                                  
            image = imSim.image(kwargs_lens=kwargs_lens_model_list,
                                kwargs_lens_light=kwargs_lens_light_list,
                                kwargs_source=kwargs_source_list,
                                kwargs_ps=kwargs_point_source_list)
                                
                
            # Add noise
            for noise_source_num in range(1, sim_dict['NUMBER_OF_NOISE_SOURCES'] + 1):
                image += self.generate_noise(sim_dict['NOISE_SOURCE_{0}-NAME'.format(noise_source_num)],
                                             np.shape(image),
                                             select_params(sim_dict, 'NOISE_SOURCE_{0}-'.format(noise_source_num)))
            
            # Combine with other bands
            output_image.append(image)
        
        # Return simulated image
        return np.array(output_image)

    def generate_noise(self, name, shape, params):
        """
        Add noise to image based on input yaml by targeting specified distribution
        
        :param name: name of the distribution to target
        :param shape: shape of image to add noise to
        :param params: dictionary of additional parameters needed by distributions.name()
        :return: noise_image: noise from targeted distribution for the image
        """
        return eval('distributions.{0}(shape, **params)'.format(name.lower()))

    

                        

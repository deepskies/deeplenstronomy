#!/usr/bin/env python
# coding: utf-8

# # LensPipe
# 
# Written by Robert Morgan
# 
# A framework for interacting with `lenstronomy` via a single yaml configuration file.

# If you need the `benedict` module, execute the following cell and restart your kernel.

# In[ ]:


#! pip install python-benedict


# In[ ]:


from benedict import benedict
import time
import yaml

import distributions


# In[ ]:


input_yaml = """

LIGHT_PROFILE_1:
    NAME: SERSIC_ELLIPSE
    PARAMETERS:
        center_x:
            DISTRIBUTION:
                NAME: uniform
                PARAMETERS:
                    minimum: -0.5
                    maximum: 0.5
        center_y:
            DISTRIBUTION:
                NAME: uniform
                PARAMETERS:
                    minimum: -0.5
                    maximum: 0.5

        R_sersic:
            DISTRIBUTION:
                NAME: normal
                PARAMETERS:
                    mean: 100
                    std: 5

        n_sersic:
            DISTRIBUTION:
                NAME: normal
                PARAMETERS:
                    mean: 100
                    std: 5

"""
with open('sersic_light_defaults.yaml', 'w+') as fake_default_file:
    fake_default_file.write(input_yaml)


# In[ ]:


fake_config = """
DATASET:
    NAME: DeepLenstronomyData
    PARAMETERS:
        SIZE: 100000
        OUTDIR: temp

IMAGE:
    PARAMETERS:
        exptime: 100.0
        fwhm: 
            DISTRIBUTION:
                NAME: uniform
                PARAMETERS:
                    minimum: 0.8
                    maximum: 1.4
        kernel_size: 91
        numPix: 101
        psf_type: 'GAUSSIAN'
        sigma_bkg: 8.0
        truncation: 3
        
COSMOLOGY:
    PARAMETERS:
        H0: 70
        Om0: 0.3

SURVEY:
    PARAMETERS:
        pixel_size: 0.263
        BANDS: g,r,i,z
        SKY_BRIGHTNESS:
            DISTRIBUTION:
                NAME: normal
                PARAMETERS:
                    mean: 30.0
                    std: 1.0

SPECIES:
    GALAXY_1:
        NAME: LENS
        LIGHT_PROFILE_1:
            NAME: SERSIC_ELLIPSE
            PARAMETERS:
                center_x:
                    DISTRIBUTION:
                        NAME: uniform
                        PARAMETERS:
                            minimum: -0.5
                            maximum: 0.5
                center_y:
                    DISTRIBUTION:
                        NAME: uniform
                        PARAMETERS:
                            minimum: -0.5
                            maximum: 0.5

                R_sersic:
                    DISTRIBUTION:
                        NAME: normal
                        PARAMETERS:
                            mean: 100
                            std: 5

                n_sersic:
                    DISTRIBUTION:
                        NAME: normal
                        PARAMETERS:
                            mean: 100
                            std: 5
                            
        MASS_PROFILE_1:
            NAME: m
            PARAMETERS:
                mass_param_1: 6
            
        SHEAR_PROFILE_1:
            NAME: s
            PARAMETERS:
                gamma_ext:
                    DISTRIBUTION: 
                        NAME: uniform
                        PARAMETERS:
                            minimum: 4
                            maximum: 5
                    
                        
            
    GALAXY_2:
        NAME: SOURCE
        INPUT: sersic_light_defaults.yaml
        
    POINTSOURCE_1:
        NAME: AGN
        HOST: SOURCE
        PARAMETERS:
            magnitude:
                DISTRIBUTION:
                    NAME: uniform
                    PARAMETERS:
                        minimum: 14
                        maximum: 23
            
    POINTSOURCE_2:
        NAME: SUPERNOVA
        HOST: LENS
        PARAMETERS:
        
    POINTSOURCE_3:
        NAME: STAR
        HOST: None
        PARAMETERS:
        
    NOISE_1:
        NAME: POISSON_NOISE
        PARAMETERS:
            mean:
                DISTRIBUTION:
                    NAME: uniform
                    PARAMETERS:
                        minimum: 1
                        maximum: 2
    NOISE_2:
        NAME: DES_NOISE
        PARAMETERS:
        
GEOMETRY:

    CONFIGURATION_1:
        NAME: GALAXY_AGN
        FRACTION: 0.5
        PLANE_1:
            OBJECT_1: LENS
            PARAMETERS:
                REDSHIFT:
                    DISTRIBUTION:
                        NAME: delta_function
                        PARAMETERS:
                            value: 0.5
                            
        PLANE_2:
            OBJECT_1: SOURCE
            OBJECT_2: AGN
            PARAMETERS:
                REDSHIFT:
                    DISTRIBUTION:
                        NAME: uniform
                        PARAMETERS:
                            minimum: 1.0
                            maximum: 1.2
                            
        NOISE_SOURCE_1: POISSON_NOISE
                            
    CONFIGURATION_2:
        FRACTION: 0.3
        NAME: GALAXY_AGN
        PLANE_1: 
            OBJECT_1: LENS
            PARAMETERS:
                REDSHIFT:
                    DISTRIBUTION:
                        NAME: delta_function
                        PARAMETERS:
                            value: 0.5
        PLANE_2:
            OBJECT_1: SOURCE
            OBJECT_2: AGN
            PARAMETERS:
                REDSHIFT:
                    DISTRIBUTION:
                        NAME: uniform
                        PARAMETERS:
                            minimum: 1.0
                            maximum: 1.2
            
            
    CONFIGURATION_3:
        NAME: DOUBLE_SOURCE_PLANE_AGN
        FRACTION: 0.2
        PLANE_1:
            OBJECT_1: LENS
            PARAMETERS: 
                REDSHIFT:
                    DISTRIBUTION:
                        NAME: delta_function
                        PARAMETERS:
                            value: 0.2
        PLANE_2:
            OBJECT_1: LENS
            PARAMETERS: 
                REDSHIFT:
                    DISTRIBUTION:
                        NAME: delta_function
                        PARAMETERS:
                            value: 0.5
        PLANE_3:
            OBJECT_1: SOURCE
            OBJECT_2: AGN
            PARAMETERS:
                REDSHIFT:
                    DISTRIBUTION:
                        NAME: delta_function
                        PARAMETERS:
                            value: 1.0
            

"""

with open('fake_config.yaml', 'w+') as fake_config_file:
    fake_config_file.write(fake_config)


# In[ ]:


class Parser():
    def __init__(self, config):
        
        # Read main configuration file
        self.full_dict = self.read(config)
        
        # If the main file points to any input files, read those too
        self.get_input_locations()
        self.include_inputs()
        
        # Check for user errors in inputs
        self.check()
        
        return
    
    def include_inputs(self):
        """
        Adds any input yaml files to config dict and assigns to self.
        """
        config_dict = benedict(self.full_dict.copy(), keypath_separator='.')
        
        for input_path in self.input_paths:
            input_dict = self.read(config_dict[input_path + '.INPUT'])
            for k, v in input_dict.items():
                config_dict[input_path + '.{0}'.format(k)] = v
                
        self.config_dict = benedict(config_dict, keypath_separator=None)
        return    
    
    def get_input_locations(self):
        """
        Find locations in main dictionary where input files are listed
        """
        d = benedict(self.full_dict, keypath_separator='.')
        input_locs = [x.find('INPUT') for x in d.keypaths()]
        input_paths = [y for y in [x[0:k-1] if k != -1 else '' for x, k in zip(d.keypaths(), input_locs)] if y != '']
        self.input_paths = input_paths
        return
        
    
    def read(self, config):
        """
        Reads config file into a dictionary and returns it.
        
        :param config: Name of config file.
        :return: config_dict: Dictionary containing config information.
        """
        with open(config, 'r') as config_file_obj:
            config_dict = yaml.safe_load(config_file_obj)
            
        return config_dict

    
    def check(self):
        """
        Check configurations for possible user errors.
        """
        
        #BIG TODO
        
        
        #don't name an object 'None'
        #object names must be unique
        #spell 'parameters' right for once in your life
        return
    


# In[ ]:


class Organizer():
    def __init__(self, config_dict):
        """
        Break up config dict into individual simulation dicts.
        
        :param config_dict: Parser.config_dict
        """
        self.main_dict = config_dict.copy()
        
        self.__track_species_keys()
        
        self.breakup()
        
        return
    
    def __track_species_keys(self):
        """Create a map of object name to species keys"""
        species_map = {}
        for k, v in self.main_dict['SPECIES'].items():
            species_map[v['NAME']] = k
        self._species_map = species_map
        return
            
    
    def _convert_to_string(self, distribution_dict, bands):
        """
        Convert distribution dict into callable method
        
        :param distribution_dict: dicitonary containing pdf info
        :return: method: callable method as string
        """
        #this some black magic
        return distribution_dict['NAME'] + '(' + ', '.join(['{0}={1}'.format(k, v) for k, v in distribution_dict['PARAMETERS'].items()]) + ', bands="{0}"'.format(','.join(bands)) + ')'
    
    def _draw(self, distribution_dict, bands):
        """
        Draw a random value from the specified distribution
        
        :param distribution_dict: dicitonary containing pdf info
        :return: value: sampled value from distribution
        """
        draw_command = 'distributions.{0}'.format(self._convert_to_string(distribution_dict, bands))
        return eval(draw_command)
    
    def _flatten_and_fill(self, config_dict):
        """
        Flatten input dictionary, and sample from any specified distributions
        
        :param config_dict: dictionary built up by self.breakup()
        :return: flattened_and_filled dictionary: dict ready for individual image sim
        """
        bands = config_dict['SURVEY_DICT']['BANDS'].split(',')
        output_dict = {x: {} for x in bands}
        
        #COSMOLOGY
        for k, v in config_dict['COSMOLOGY_DICT'].items():
            if v != 'DISTRIBUTION':
                for band in bands:
                    output_dict[band][k] = v
            else:
                draws = self._draw(self.main_dict['COSMOLOGY']['PARAMETERS'][k]['DISTRIBUTION'], bands)
                for band, draw in zip(bands, draws):
                    output_dict[band][k] = draw

        #IMAGE
        for k, v in config_dict['IMAGE_DICT'].items():
            if v != 'DISTRIBUTION':
                for band in bands:
                    output_dict[band][k] = v
            else:
                draws = self._draw(self.main_dict['IMAGE']['PARAMETERS'][k]['DISTRIBUTION'], bands)
                for band, draw in zip(bands, draws):
                    output_dict[band][k] = draw

        #SURVEY
        for k, v in config_dict['SURVEY_DICT'].items():
            if k == 'BANDS': 
                continue
            if v != 'DISTRIBUTION':
                for band in bands:
                    output_dict[band][k] = v
            else:
                draws = self._draw(self.main_dict['SURVEY']['PARAMETERS'][k]['DISTRIBUTION'], bands)
                for band, draw in zip(bands, draws):
                    output_dict[band][k] = draw
                    
        #NOISE
        for noise_idx in range(config_dict['NOISE_DICT']['NUMBER_OF_NOISE_SOURCES']):
            noise_source_num = noise_idx + 1
            noise_name = config_dict['NOISE_DICT']['NOISE_SOURCE_{0}-NAME'.format(noise_source_num)]
            for k, v in self.main_dict['SPECIES'][self._species_map[noise_name]]['PARAMETERS'].items():
                if isinstance(v, dict):
                    draws = self._draw(v['DISTRIBUTION'], bands)
                    for band, draw in zip(bands, draws):
                        output_dict[band]['NOISE_SOURCE_{0}-{1}'.format(noise_source_num, k)] = draw
                else:
                    for band in bands:
                        output_dict[band]['NOISE_SOURCE_{0}-{1}'.format(noise_source_num, k)] = v


        #REAL OBJECTS
        for k, v in config_dict['SIM_DICT'].items():
            for band in bands:
                output_dict[band][k] = v

        for plane_idx in range(config_dict['SIM_DICT']['NUMBER_OF_PLANES']):
            geometry_key = config_dict['SIM_DICT']['CONFIGURATION_LABEL']
            plane_num = plane_idx + 1
            for obj_idx in range(config_dict['SIM_DICT']['PLANE_{0}-NUMBER_OF_OBJECTS'.format(plane_num)]):
                obj_num = obj_idx + 1
                obj_name = config_dict['SIM_DICT']['PLANE_{0}-OBJECT_{1}-NAME'.format(plane_num, obj_num)]

                #GEOMETRY
                for k_param, v_param in self.main_dict['GEOMETRY'][geometry_key]['PLANE_{0}'.format(plane_num)]['PARAMETERS'].items():
                    if isinstance(v_param, dict):
                        draws = self._draw(v_param['DISTRIBUTION'], bands)
                        for band, draw in zip(bands, draws):
                            output_dict[band]['PLANE_{0}-OBJECT_{1}-{2}'.format(plane_num, obj_num, k_param)] = draw
                    else:
                        for band in bands:
                            output_dict[band]['PLANE_{0}-OBJECT_{1}-{2}'.format(plane_num, obj_num, k_param)] = v_param
                            
                #SPECIES- Hosts
                if 'HOST' in self.main_dict['SPECIES'][self._species_map[obj_name]].keys():
                    output_dict[band]['PLANE_{0}-OBJECT_{1}-HOST'.format(plane_num, obj_num)] = self.main_dict['SPECIES'][self._species_map[obj_name]]['HOST']
                else:
                    output_dict[band]['PLANE_{0}-OBJECT_{1}-HOST'.format(plane_num, obj_num)] = 'None'
                    
                #SPECIES- Light Profiles
                for light_profile_idx in range(config_dict['SPECIES_DICT'][obj_name]['NUMBER_OF_LIGHT_PROFILES']):
                    light_profile_num = light_profile_idx + 1
                    output_dict[band]['PLANE_{0}-OBJECT_{1}-LIGHT_PROFILE_{2}-NAME'.format(plane_num, obj_num, light_profile_num)] = self.main_dict['SPECIES'][self._species_map[obj_name]]['LIGHT_PROFILE_{0}'.format(light_profile_num)]['NAME']
                    for k_param, v_param in self.main_dict['SPECIES'][self._species_map[obj_name]]['LIGHT_PROFILE_{0}'.format(light_profile_num)]['PARAMETERS'].items():
                        if isinstance(v_param, dict):
                            draws = self._draw(v_param['DISTRIBUTION'], bands)
                            for band, draw in zip(bands, draws):
                                output_dict[band]['PLANE_{0}-OBJECT_{1}-LIGHT_PROFILE_{2}-{3}'.format(plane_num, obj_num, light_profile_num, k_param)] = draw
                        else:
                            for band in bands:
                                output_dict[band]['PLANE_{0}-OBJECT_{1}-LIGHT_PROFILE_{2}-{3}'.format(plane_num, obj_num, light_profile_num, k_param)] = v_param
                
                #SPECIES- Mass Profiles
                for mass_profile_idx in range(config_dict['SPECIES_DICT'][obj_name]['NUMBER_OF_MASS_PROFILES']):
                    mass_profile_num = mass_profile_idx + 1
                    output_dict[band]['PLANE_{0}-OBJECT_{1}-MASS_PROFILE_{2}-NAME'.format(plane_num, obj_num, mass_profile_num)] = self.main_dict['SPECIES'][self._species_map[obj_name]]['MASS_PROFILE_{0}'.format(mass_profile_num)]['NAME']
                    for k_param, v_param in self.main_dict['SPECIES'][self._species_map[obj_name]]['MASS_PROFILE_{0}'.format(mass_profile_num)]['PARAMETERS'].items():
                        if isinstance(v_param, dict):
                            draws = self._draw(v_param['DISTRIBUTION'], bands)
                            for band, draw in zip(bands, draws):
                                output_dict[band]['PLANE_{0}-OBJECT_{1}-MASS_PROFILE_{2}-{3}'.format(plane_num, obj_num, mass_profile_num, k_param)] = draw
                        else:
                            for band in bands:
                                output_dict[band]['PLANE_{0}-OBJECT_{1}-MASS_PROFILE_{2}-{3}'.format(plane_num, obj_num, mass_profile_num, k_param)] = v_param
                #SPECIES- Shear Profiles
                for shear_profile_idx in range(config_dict['SPECIES_DICT'][obj_name]['NUMBER_OF_SHEAR_PROFILES']):
                    shear_profile_num = shear_profile_idx + 1
                    output_dict[band]['PLANE_{0}-OBJECT_{1}-SHEAR_PROFILE_{2}-NAME'.format(plane_num, obj_num, shear_profile_num)] = self.main_dict['SPECIES'][self._species_map[obj_name]]['SHEAR_PROFILE_{0}'.format(shear_profile_num)]['NAME']
                    for k_param, v_param in self.main_dict['SPECIES'][self._species_map[obj_name]]['SHEAR_PROFILE_{0}'.format(shear_profile_num)]['PARAMETERS'].items():
                        if isinstance(v_param, dict):
                            draws = self._draw(v_param['DISTRIBUTION'], bands)
                            for band, draw in zip(bands, draws):
                                output_dict[band]['PLANE_{0}-OBJECT_{1}-SHEAR_PROFILE_{2}-{3}'.format(plane_num, obj_num, shear_profile_num, k_param)] = draw
                        else:
                            for band in bands:
                                output_dict[band]['PLANE_{0}-OBJECT_{1}-SHEAR_PROFILE_{2}-{3}'.format(plane_num, obj_num, shear_profile_num, k_param)] = v_param

                #SPECIES- Additional Parameters
                if 'PARAMETERS' in self.main_dict['SPECIES'][self._species_map[obj_name]].keys():
                    for k_param, v_param in self.main_dict['SPECIES'][self._species_map[obj_name]]['PARAMETERS'].items():
                        if isinstance(v_param, dict):
                            draws = self._draw(v_param['DISTRIBUTION'], bands)
                            for band, draw in zip(bands, draws):
                                output_dict[band]['PLANE_{0}-OBJECT_{1}-{2}'.format(plane_num, obj_num, k_param)] = draw
                        else:
                            for band in bands:
                                output_dict[band]['PLANE_{0}-OBJECT_{1}-{2}'.format(plane_num, obj_num, k_param)] = v_param

        return output_dict
        
    def breakup(self):
        """
        Based on configurations and dataset size, build list of simulation dicts.
        """
        # Determine number of images to simulate for each configuration
        global_size = self.main_dict['DATASET']['PARAMETERS']['SIZE']
        configurations = {}
        for k, v in self.main_dict['GEOMETRY'].items():
            configurations[k] = v
            configurations[k]['SIZE'] = int(global_size * v['FRACTION'])
        
        # Determine objects and their planes, store in SIM_DICT key
        for k, v in configurations.items():
            sim_dict = {}
            
            sim_dict['CONFIGURATION_LABEL'] = k
            sim_dict['CONFIGURATION_NAME'] = v['NAME']
            sim_dict['NUMBER_OF_PLANES'] = len([x for x in v.keys() if x.find('PLANE') != -1])
            
            for config_key, config_dict in v.items():
                if config_key.find('PLANE') != -1:
                    sim_dict['PLANE_{0}-NUMBER_OF_OBJECTS'.format(config_key.split('_')[-1])] = len([y for y in config_dict.keys() if y.find('OBJECT') != -1])
                    for obj_index in [x.split('_')[-1] for x in [y for y in config_dict.keys() if y.find('OBJECT') != -1]]:
                        sim_dict[config_key + '-' + 'OBJECT_{0}-NAME'.format(obj_index)] = config_dict['OBJECT_{0}'.format(obj_index)]
                    
            configurations[k]['SIM_DICT'] = sim_dict
            configurations[k] = dict_select(configurations[k], ['NAME', 'SIZE', 'SIM_DICT'])
            
        # Determine number of profiles for each object
        species_dict = {}
        for k, v in self.main_dict['SPECIES'].items():
            species_dict[v['NAME']] = {'NUMBER_OF_LIGHT_PROFILES': len([x for x in v.keys() if x.find('LIGHT_PROFILE') != -1]),
                                       'NUMBER_OF_MASS_PROFILES': len([x for x in v.keys() if x.find('MASS_PROFILE') != -1]),
                                       'NUMBER_OF_SHEAR_PROFILES': len([x for x in v.keys() if x.find('SHEAR_PROFILE') != -1])}
        for k in configurations.keys():
            configurations[k]['SPECIES_DICT'] = species_dict

        # Add image metadata
        image_dict = {k: v if not isinstance(v, dict) else 'DISTRIBUTION' for k, v in self.main_dict['IMAGE']['PARAMETERS'].items()}
        for k in configurations.keys():
            configurations[k]['IMAGE_DICT'] = image_dict
        
        # Add survey metadata
        survey_dict = {k: v if not isinstance(v, dict) else 'DISTRIBUTION' for k, v in self.main_dict['SURVEY']['PARAMETERS'].items()}
        #survey_dict['NAME'] = self.main_dict['SURVEY']['NAME']
        for k in configurations.keys():
            configurations[k]['SURVEY_DICT'] = survey_dict
            
        # Add cosmology metadata
        cosmo_dict = {k: v if not isinstance(v, dict) else 'DISTRIBUTION' for k, v in self.main_dict['COSMOLOGY']['PARAMETERS'].items()}
        #cosmo_dict['NAME'] = self.main_dict['COSMOLOGY']['NAME']
        for k in configurations.keys():
            configurations[k]['COSMOLOGY_DICT'] = cosmo_dict
            
        # Add noise metadata
        for k in configurations.keys():
            noise_dict = {}
            number_of_noise_sources = len([x for x in self.main_dict['GEOMETRY'][k].keys() if x.find('NOISE_SOURCE') != -1])
            noise_dict['NUMBER_OF_NOISE_SOURCES'] = number_of_noise_sources
            for noise_source_idx in range(number_of_noise_sources):
                noise_source_num = noise_source_idx + 1
                noise_dict['NOISE_SOURCE_{0}-NAME'.format(noise_source_num)] = self.main_dict['GEOMETRY'][k]['NOISE_SOURCE_{0}'.format(noise_source_num)]
            configurations[k]['NOISE_DICT'] = noise_dict
        
        # For each configuration, generate full sim info for as many objects as user specified
        configuration_sim_dicts = {}
        for k, v in configurations.items():
            configuration_sim_dicts[k] = []
            
            for _ in range(v['SIZE']):
                configuration_sim_dicts[k].append(self._flatten_and_fill(v.copy()))
                
        self.configuration_sim_dicts = configuration_sim_dicts


# In[ ]:


from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.SimulationAPI.sim_api import SimAPI


# In[ ]:


def dict_select(input_dict, keys):
    """
    Trim a dictionary down to selected keys
    
    :param input_dict: full dictionary
    :param keys: list of keys desired in trimmed dict
    :return: dict: trimmed dictionary
    """
    return {k: input_dict[k] for k in keys}

class ImageGenerator():
    def __init__(self):
        return
    
    def sim_image(info_dict):
        """
        Simulate an image based on specifications in sim_dict
        
        :param info_dict: A single element of the output form Organizer.breakup()
        """
        
        for band, sim_dict in info_dict.items():
            # PSF
            psf_info = ['psf_type', 'fwhm', 'pixel_size', 'truncation']
            psf_class = PSF(**dict_select(sim_dict, psf_info))
        
            # Image Data
            image_info = ['numPix', 'deltaPix', 'exp_time', 'sigma_bkg']
            image_class = ImageData(**dict_select(sim_dict, image_info))
        
        
        


# In[ ]:


# This cell will take a couple minutes to run, the run time scales linearly with the dataset size
start = time.time()
P = Parser('fake_config.yaml')
middle = time.time()
O = Organizer(P.config_dict)
end = time.time()

print("Parse Time:    ", round(middle - start, 4), "sec")
print("Organize Time: ", round(end - middle, 4), "sec")
print("Total Time:    ", round(end - start, 4), "sec")


# ## Demo
# 
# Dictionaries for all events to be simulated have been stored in `O.configuration_sim_dicts`

# In[ ]:


print([x for x in dir(O) if x[0:2] != '__'])


# In[ ]:


print(O.configuration_sim_dicts.keys())


# In[ ]:


print(type(O.configuration_sim_dicts['CONFIGURATION_1']))


# In[ ]:


print(len(O.configuration_sim_dicts['CONFIGURATION_1']))


# In[ ]:


print(O.configuration_sim_dicts['CONFIGURATION_1'][0])


# ## TODO
# 
# 1. Complete the image generator class to go from the configuration dictionaries to a set of images
# 2. Write the `Parser.check()` method to look for potential errors in the input yaml file
# 3. Create a set of standard yaml files for usage
# 

# In[ ]:





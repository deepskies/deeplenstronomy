"""Parse a user configuration file."""

import copy
import random
import os
import sys
import yaml

from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Analysis.td_cosmography import TDCosmography
from lenstronomy.SimulationAPI.sim_api import SimAPI
import numpy as np
import pandas as pd

import deeplenstronomy.timeseries as timeseries
from deeplenstronomy.utils import dict_select, dict_select_choose, draw_from_user_dist, KeyPathDict, read_cadence_file
import deeplenstronomy.distributions as distributions
import deeplenstronomy.special as special
import deeplenstronomy.surveys as surveys
import deeplenstronomy.check as big_check
import deeplenstronomy.image_generator as image_generator

class Parser():
    """ 
    Load yaml inputs into a single dictionary and trigger automatic checks for user errors.

    """

    def __init__(self, config, survey=None):
        """
        Args: 
            config (str): name of yaml configuration file
            survey (str or None, optional, default=None): Automatically passed from deeplenstronomy.make_dataset() args
        """
        
        # Check for annoying tabs - there's probably a better way to do this
        self._parse_for_tabs(config)

        # Fill in sections of the configuration file for a specific survey
        if survey is not None:
            config = self.write_survey(config, survey)
        
        # Read main configuration file
        self.full_dict = self.read(config)
        
        # If the main file points to any input files, read those too
        self._get_input_locations()
        self._include_inputs()

        # Check for user-specifed probability distributions and backgrounds
        self._get_file_locations()
        self._get_image_locations()
        
        # Check for user errors in inputs
        self.check()

        return


    def write_survey(self, config, survey):
        """
        Writes survey information to config file. Creates a new file named {survey}_{config}
        by copying the contents of {config} and appending the IMAGE and SURVEY sections for 
        a desired survey. The yaml parser will automatically overwrite the IMAGE and SURVEY
        dictionary keys.

        Args:
            config (str): name of yaml configuration file
            survey (str or None, optional, default=None): Automatically passed from deeplenstronomy.make_dataset() args

        Returns:
            outfile (str): name of survey-specific configuration file
        """
        # set new config file name
        config_basename = config.split('/')
        if len(config_basename) == 1:
            outfile = survey + '_' + config
        else:
            outfile = '/'.join(config_basename[0:-1]) + '/' + survey + '_' + config_basename[-1]

        # write new config file
        with open(config, 'r') as old, open(outfile, 'w+') as new:
            new.writelines(old.readlines())
            new.writelines(eval("surveys.{}()".format(survey)))
        return outfile
    
    def _include_inputs(self):
        """
        Searches for uses of the keyword INPUT and adds the file contents to the main configuration dictionary.
        """
        config_dict = KeyPathDict(self.full_dict.copy(), keypath_separator='.')
        
        for input_path in self.input_paths:
            input_dict = self.read(eval('config_dict["' + input_path.replace('.', '"]["') + '"]["INPUT"]'))
            for k, v in input_dict.items():
                exec('config_dict["' + input_path.replace('.', '"]["') + '"][k] = v')

        self.config_dict = config_dict
        return    

    def _get_input_locations(self):
        input_paths = self._get_kw_locations("INPUT")
        self.input_paths = input_paths
        return

    def _get_file_locations(self):        
        file_paths = []
        if "DISTRIBUTIONS" in self.full_dict.keys():
            for k in self.full_dict['DISTRIBUTIONS'].keys():
                file_paths.append('DISTRIBUTIONS.' + k)
        self.file_paths = file_paths

        return

    def _get_image_locations(self):
        file_paths = []
        image_configurations = []
        if "BACKGROUNDS" in self.full_dict.keys():
            file_paths.append(self.full_dict['BACKGROUNDS']['PATH'])
            self.image_configurations = self.full_dict['BACKGROUNDS']['CONFIGURATIONS'][:]
        self.image_paths = file_paths

        return
    
    def _get_kw_locations(self, kw):
        """
        Find locations in main dictionary where a keyword is used

        :param kw: str, a keyword to search the dict keys for
        :return: paths: list, the keypaths to all occurances of kw
        """
        d = KeyPathDict(self.full_dict, keypath_separator='.')
        locs = [x.find(kw) for x in d.keypaths()]
        paths = [y for y in [x[0:k-1] if k != -1 else '' for x, k in zip(d.keypaths(), locs)] if y != '']
        return paths


    def read(self, config):
        """
        Reads config file into a dictionary and returns it.
        
        Args:
            config (str): Name of config file.
        
        Returns:
            config_dict (dict): Dictionary containing config information.
        """

        with open(config, 'r') as config_file_obj:
            config_dict = yaml.safe_load(config_file_obj)
                        
        return config_dict


    def _parse_for_tabs(self, config):
        """
        Check for the existence of tab characters that might break yaml
        """
        stream = open(config, 'r')
        lines = stream.readlines()
        stream.close()

        bad_line_numbers = []
        for index, line in enumerate(lines):
            if line.find('\t') != -1:
                bad_line_numbers.append(str(index + 1))

        if len(bad_line_numbers) != 0:
            print("Tab characters detected on the following lines:")
            print("    " + ', '.join(bad_line_numbers))
            print("Please correct the tabs and restart")
            sys.exit()
        return
    
    def check(self):
        """
        Check configuration file for possible user errors.
        """
        big_check._run_checks(self.full_dict, self.config_dict)
        
        return
    

class Organizer():
    def __init__(self, config_dict, forced_inputs={}, verbose=False):
        """
        Break up config dict into individual simulation dicts.
        
        Args:
            config_dict (dict): an instance of Parser.config_dict
            verbose (bool, optional, default=False): Automatically passed from deeplenstronomy.make_dataset() args
        """
        self.main_dict = config_dict.copy()
        self.forced_inputs = forced_inputs
        
        self.__track_species_keys()
        
        self.breakup(verbose=verbose)
        
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
        #this some magic
        if isinstance(distribution_dict['PARAMETERS'], dict):
            return distribution_dict['NAME'] + '(' + ', '.join(['{0}={1}'.format(k, v) for k, v in distribution_dict['PARAMETERS'].items()]) + ', bands="{0}"'.format(','.join(bands)) + ')'
        else:
            return distribution_dict['NAME'] + '(bands="{0}")'.format(','.join(bands))
        

    def _draw(self, distribution_dict, bands):
        """
        Draw a random value from the specified distribution
        
        :param distribution_dict: dicitonary containing pdf info
        :return: value: sampled value from distribution
        """
        draw_command = 'distributions.{0}'.format(self._convert_to_string(distribution_dict, bands))
        return eval(draw_command)

    def _choose_position(self, ra_host, dec_host, sep, sep_unit, cosmo, redshift=None, angle=None):
        """
        Select an ra and dec that will be sep away from the host
        
        :param ra_host: x-coord of point source host
        :param dec_host: y-coord of point source host
        :param sep: angular separation between point source and host
        :param sep_unit: either 'kpc' or 'arcsec'
        :param redshift: cosmological redshift, required if units are in kpc
        :param angle: desired position of point source in radians, random if None
        :return: chosen_ra: x-coord of chosen point sep away from host
        :return: chosen_dec: y-coord of chosen point sep away from host
        """
        if angle is None:
            angle = random.uniform(0.0, 2 * np.pi)

        if sep_unit == 'arcsec':            
            chosen_ra = np.cos(angle) * sep + ra_host
            chosen_dec = np.sin(angle) * sep + dec_host
        elif sep_unit == 'kpc':
            kpc_to_arcsec = cosmo.arcsec_per_kpc_comoving(redshift).value / (1. + redshift)
            chosen_ra = np.cos(angle) * sep * kpc_to_arcsec + ra_host
            chosen_dec = np.sin(angle) * sep * kpc_to_arcsec + dec_host
        else:
            raise NotImplementedError("unexpected sep_unit")
        
        return chosen_ra, chosen_dec

    def _find_obj_string(self, obj_name, configuration):
        """
        Return the location of an object in the flattened dictionary
        
        :param obj_name: the name of the object
        :param configuration: 'CONFIGURATION_1', 'CONFIGURATION_2', etc.
        :return: obj_string: the location of the object in the flattened dictionary
        """

        d = KeyPathDict(self.main_dict['GEOMETRY'][configuration].copy(), keypath_separator='.')
        for x in d.keypaths():
            f = "['" + "']['".join(x.split('.')) + "']"
            k = eval("d" + f)
            if k == obj_name:
                return x.replace('.', '-')

        #return [x.replace('.', '-') for x in d.keypaths() if eval("d['" + "']['".join(x.split('.')) + "']") == obj_name][0]

    
    def _flatten_and_fill(self, config_dict, cosmo, inputs, objid=0):
        """
        Flatten input dictionary, and sample from any specified distributions
        
        :param config_dict: dictionary built up by self.breakup()
        :param cosmo: an astropy.cosmology instance
        :return: flattened_and_filled dictionary: dict ready for individual image sim
        """
        bands = config_dict['SURVEY_DICT']['BANDS'].split(',')
        output_dict = {x: {} for x in bands}

        #Object IDs
        for band in bands:
            output_dict[band]['OBJID'] = objid

        #Pointing - Timeseries only
        if hasattr(self, "cadence_dict"):
            pointing = random.choice(list(set(self.cadence_dict.keys()) - set(['REFERENCE_MJD'])))
            for band in bands:
                output_dict[band]['POINTING'] = pointing
        
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
        for band in bands:
            output_dict[band]['NUMBER_OF_NOISE_SOURCES'] = config_dict['NOISE_DICT']['NUMBER_OF_NOISE_SOURCES']
        for noise_idx in range(config_dict['NOISE_DICT']['NUMBER_OF_NOISE_SOURCES']):
            noise_source_num = noise_idx + 1
            noise_name = config_dict['NOISE_DICT']['NOISE_SOURCE_{0}-NAME'.format(noise_source_num)]
            for band in bands:
                output_dict[band]['NOISE_SOURCE_{0}-NAME'.format(noise_source_num)] = noise_name
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

            #GEOMETRY
            for k_param, v_param in self.main_dict['GEOMETRY'][geometry_key]['PLANE_{0}'.format(plane_num)]['PARAMETERS'].items():
                if isinstance(v_param, dict):
                    draws = self._draw(v_param['DISTRIBUTION'], bands)

                    # Set the PLANE's redshift in the config_dict
                    if k_param == 'REDSHIFT':
                        config_dict['SIM_DICT']['PLANE_{0}-REDSHIFT'.format(plane_num)] = draws[0]
                    
                    for band, draw in zip(bands, draws):
                        for obj_num in range(1, config_dict['SIM_DICT']['PLANE_{0}-NUMBER_OF_OBJECTS'.format(plane_num)] + 1):
                            output_dict[band]['PLANE_{0}-OBJECT_{1}-{2}'.format(plane_num, obj_num, k_param)] = draw
                else:
                    # Set the PLANE's redshift in the config_dict
                    if k_param == 'REDSHIFT':
                        config_dict['SIM_DICT']['PLANE_{0}-REDSHIFT'.format(plane_num)]	= v_param
                    
                    for band in bands:
                        for obj_num in range(1, config_dict['SIM_DICT']['PLANE_{0}-NUMBER_OF_OBJECTS'.format(plane_num)] + 1):
                            output_dict[band]['PLANE_{0}-OBJECT_{1}-{2}'.format(plane_num, obj_num, k_param)] = v_param

            for obj_idx in range(config_dict['SIM_DICT']['PLANE_{0}-NUMBER_OF_OBJECTS'.format(plane_num)]):
                obj_num = obj_idx + 1
                obj_name = config_dict['SIM_DICT']['PLANE_{0}-OBJECT_{1}-NAME'.format(plane_num, obj_num)]
                
                #save number of profiles
                for band in bands:
                    output_dict[band]['PLANE_{0}-OBJECT_{1}-NUMBER_OF_LIGHT_PROFILES'.format(plane_num, obj_num)] = config_dict['SPECIES_DICT'][obj_name]['NUMBER_OF_LIGHT_PROFILES']
                    output_dict[band]['PLANE_{0}-OBJECT_{1}-NUMBER_OF_SHEAR_PROFILES'.format(plane_num, obj_num)] = config_dict['SPECIES_DICT'][obj_name]['NUMBER_OF_SHEAR_PROFILES']
                    output_dict[band]['PLANE_{0}-OBJECT_{1}-NUMBER_OF_MASS_PROFILES'.format(plane_num, obj_num)] = config_dict['SPECIES_DICT'][obj_name]['NUMBER_OF_MASS_PROFILES']

                #SPECIES- Point Sources
                if 'HOST' in self.main_dict['SPECIES'][self._species_map[obj_name]].keys():
                    host = self.main_dict['SPECIES'][self._species_map[obj_name]]['HOST']
                    if host != 'Foreground':
                        # Get host center
                        possible_hostids = ['PLANE_{0}-OBJECT_{1}-NAME'.format(plane_num, x) for x in range(1, config_dict['SIM_DICT']['PLANE_{0}-NUMBER_OF_OBJECTS'.format(plane_num)] + 1)]
                        hostid = [x[0:-5] for x in possible_hostids if config_dict['SIM_DICT'][x] == host][0]
                        ra_host, dec_host = output_dict[bands[0]][hostid + '-LIGHT_PROFILE_1-center_x'], output_dict[bands[0]][hostid + '-LIGHT_PROFILE_1-center_y']
                        
                        # Determine location of point source in image
                        if 'sep' in self.main_dict['SPECIES'][self._species_map[obj_name]]['PARAMETERS'].keys():
                            sep_unit = self.main_dict['SPECIES'][self._species_map[obj_name]]['PARAMETERS']['sep_unit']
                            if isinstance(self.main_dict['SPECIES'][self._species_map[obj_name]]['PARAMETERS']['sep'], dict):
                                draws = self._draw(self.main_dict['SPECIES'][self._species_map[obj_name]]['PARAMETERS']['sep']['DISTRIBUTION'], bands)
                                sep = draws[0]
                            else:
                                sep = self.main_dict['SPECIES'][self._species_map[obj_name]]['PARAMETERS']['sep']

                            if 'angle' in self.main_dict['SPECIES'][self._species_map[obj_name]]['PARAMETERS'].keys():    
                                if isinstance(self.main_dict['SPECIES'][self._species_map[obj_name]]['PARAMETERS']['angle'], dict):
                                    draws = self._draw(self.main_dict['SPECIES'][self._species_map[obj_name]]['PARAMETERS']['angle']['DISTRIBUTION'], bands)
                                    angle = draws[0]
                                else:
                                    angle = self.main_dict['SPECIES'][self._species_map[obj_name]]['PARAMETERS']['angle']
                            else:
                                angle = None
                                
                            ##convert image separation into ra and dec
                            ra, dec = self._choose_position(ra_host, dec_host, sep, sep_unit, cosmo, config_dict['SIM_DICT']['PLANE_{0}-REDSHIFT'.format(plane_num)], angle)

                        else:
                            #set ra and dec to host center
                            ra, dec = ra_host, dec_host
                            sep = 0.0
                            sep_unit = 'arcsec'

                        for band in bands:
                            output_dict[band]['PLANE_{0}-OBJECT_{1}-HOST'.format(plane_num, obj_num)] = self.main_dict['SPECIES'][self._species_map[obj_name]]['HOST']
                            output_dict[band]['PLANE_{0}-OBJECT_{1}-NAME'.format(plane_num, obj_num)] = obj_name
                            output_dict[band]['PLANE_{0}-OBJECT_{1}-ra'.format(plane_num, obj_num)] = ra
                            output_dict[band]['PLANE_{0}-OBJECT_{1}-dec'.format(plane_num, obj_num)] = dec
                            output_dict[band]['PLANE_{0}-OBJECT_{1}-sep'.format(plane_num, obj_num)] = sep
                            output_dict[band]['PLANE_{0}-OBJECT_{1}-sep_unit'.format(plane_num, obj_num)] = sep_unit
                    else:
                        #foreground, choose position randomly
                        im_size = self.main_dict['IMAGE']['PARAMETERS']['numPix'] * self.main_dict['IMAGE']['PARAMETERS']['pixel_scale'] / 2
                        ra, dec = random.uniform(-1 * im_size, im_size), random.uniform(-1 * im_size, im_size)
                        if isinstance(self.main_dict['SPECIES'][self._species_map[obj_name]]['PARAMETERS']['magnitude'], dict):
                            draws = self._draw(self.main_dict['SPECIES'][self._species_map[obj_name]]['PARAMETERS']['magnitude']['DISTRIBUTION'], bands)
                        else:
                            draws = [self.main_dict['SPECIES'][self._species_map[obj_name]]['PARAMETERS']['magnitude']] * len(bands)
                        for band, magnitude in zip(bands, draws):
                            output_dict[band]['PLANE_{0}-OBJECT_{1}-HOST'.format(plane_num, obj_num)] = 'Foreground'
                            output_dict[band]['PLANE_{0}-OBJECT_{1}-NAME'.format(plane_num, obj_num)] = obj_name
                            output_dict[band]['PLANE_{0}-OBJECT_{1}-ra_image'.format(plane_num, obj_num)] = ra
                            output_dict[band]['PLANE_{0}-OBJECT_{1}-dec_image'.format(plane_num, obj_num)] = dec
                            output_dict[band]['PLANE_{0}-OBJECT_{1}-magnitude'.format(plane_num, obj_num)] = magnitude
                        
                        
                else:
                     for band in bands:
                        output_dict[band]['PLANE_{0}-OBJECT_{1}-HOST'.format(plane_num, obj_num)] = 'None'

                #SPECIES- Light Profiles
                for light_profile_idx in range(config_dict['SPECIES_DICT'][obj_name]['NUMBER_OF_LIGHT_PROFILES']):
                    light_profile_num = light_profile_idx + 1
                    for band in bands:
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
                    for band in bands:
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
                    for band in bands:
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
                        if k_param == 'sep':
                            #sampling for point source separation is already done, so don't overwrite it
                            continue
                        if isinstance(v_param, dict):
                            draws = self._draw(v_param['DISTRIBUTION'], bands)
                            for band, draw in zip(bands, draws):
                                output_dict[band]['PLANE_{0}-OBJECT_{1}-{2}'.format(plane_num, obj_num, k_param)] = draw
                        else:
                            for band in bands:
                                output_dict[band]['PLANE_{0}-OBJECT_{1}-{2}'.format(plane_num, obj_num, k_param)] = v_param

                #SPECIES- Special
                if 'SPECIAL' in self.main_dict['SPECIES'][self._species_map[obj_name]].keys():
                    for mode, args in self.main_dict['SPECIES'][self._species_map[obj_name]]['SPECIAL'].items():
                        for arg in args:
                            output_dict = eval('special.{0}(output_dict, "{1}", bands=bands)'.format(mode.lower(), arg))


        # Overwrite with any forced param inputs from USERDISTs
        if inputs is not None:
            
            for (param_name, band) in inputs.index.values:
            
                if param_name in output_dict[band]:
                    output_dict[band][param_name] = inputs[(param_name, band)]
                else:
                    print("WARNING: " + param_name + " is not present in the simulated dataset and may produce unexpected behavior. Use dataset.search(<param name>) to find all expected names")


#        for force_param, values in self.forced_inputs.items():
#            configuration, param_name, band = force_param
#            if configuration == config_dict['SIM_DICT']['CONFIGURATION_LABEL']:

#                if param_name in output_dict[band]:
#                    output_dict[band][param_name] = random.choice(values)
#                else:
#                    print("WARNING: " + param_name + " is not present in the simulated dataset and may produce unexpected behavior. Use dataset.search(<param name>) to find all expected names")
                
#                warned = False

#                for sim_input, val in zip(output_dict, values):
#                    if param_name in sim_input[band].keys():
#                        sim_input[band][param_name] = val
#                    else:
#                        if not warned:
#                            print("WARNING: " + param_name + " is not present in the simulated dataset and may produce unexpected behavior. Use dataset.search(<param name>) to find all expected names")
#                            warned = True
                            
        return output_dict


    def _flatten_and_fill_time_series(self, config_dict, cosmo, configuration, obj_strings, objid, peakshift, inputs):
        """
        Generate an image info dictionary for each step in the time series

        :param config_dict: dictionary built up by self.breakup()
        :param configuration: CONFIGURATION_1, CONFIGURATION_2, etc.
        :param obj_string: list of the strings targetting the object in the flattened dictionary (e.g. ['PLANE_2-OBJECT_2'])
        :param peakshifts: int or float in units of NITES to shift the peak
        :return: flattened_and_filled dictionary: dict ready for individual image sim  
        """
        
        output_dicts = []
        bands = self.main_dict['SURVEY']['PARAMETERS']['BANDS'].split(',')
        # Get flattened and filled dictionary
        base_output_dict = self._flatten_and_fill(config_dict, cosmo, inputs, objid)

        # Model the lens for time delay calculations
        td_dict = {}
        for obj_string in obj_strings:
            td_dict[obj_string] = None
            plane_num = int(obj_string.split('_')[1].split('-')[0])
            if plane_num >= 2:
                im_gen = image_generator.ImageGenerator(self.main_dict['SURVEY']['PARAMETERS']['BANDS'])
                params = im_gen.parse_single_band_info_dict(base_output_dict[bands[0]], cosmo)
                kwargs_lens_model_list = params[6]
                kwargs_model = params[1]
                kwargs_point_source_list = params[5] 
                # use sim API in case sigma_v is used for mass profiles
                sim = SimAPI(numpix=self.main_dict['IMAGE']['PARAMETERS']['numPix'],
                         kwargs_single_band=params[0],
                         kwargs_model=kwargs_model)
                kwargs_lens_model_list = sim.physical2lensing_conversion(kwargs_mass=kwargs_lens_model_list)

                try:
                    z_lens = kwargs_model['lens_redshift_list'][0]
                    z_source = kwargs_model['z_source']
                    td_cosmo = TDCosmography(z_lens, z_source, kwargs_model, cosmo_fiducial=cosmo)
                    td_dict[obj_string] = td_cosmo.time_delays(kwargs_lens_model_list, kwargs_point_source_list, kappa_ext=0).round().astype(int)
                except IndexError:
                    pass
        
        pointing = base_output_dict[bands[0]]['POINTING']
        closest_redshift_lcs = []
        for obj_name, obj_string in zip(self.main_dict['GEOMETRY'][configuration]['TIMESERIES']['OBJECTS'], obj_strings):
            # determine closest lc in library to redshift
            redshift = base_output_dict[bands[0]][obj_string + '-REDSHIFT']
            lcs = eval('self.{0}_{1}_lightcurves_{2}'.format(configuration, obj_name, pointing))
            closest_redshift_lcs.append(lcs['library'][np.argmin(np.abs(redshift - lcs['redshifts']))])
            
        # overwrite the image sim dictionary
        nite_dict = self.cadence_dict[pointing]
        for nite_idx in range(len(nite_dict[bands[0]])):
            for band in bands:
                orig_nite = nite_dict[band][nite_idx]
                #for orig_nite in nite_dict[band]:
                nite_ = orig_nite - peakshift
                output_dict = base_output_dict.copy()
                for obj_sting, closest_redshift_lc in zip(obj_strings, closest_redshift_lcs):

                    # account for time delay
                    td_shift = td_dict[obj_string]
                    if td_shift is None:
                        shifted_nites = [nite_]
                    else:
                        shifted_nites = [nite_] + [nite_ + x for x in list(td_shift[1:] - td_shift[0])]

                    for idx, nite in enumerate(shifted_nites):

                        if idx == 0:
                            suffix = ''
                        else:
                            suffix = f"_shift_{idx}"

                        output_dict[band][obj_string + '-tdshift_' + str(idx)] = nite
                            
                        try:
                            #try using the exact night
                            output_dict[band][obj_string + '-magnitude' + suffix] = closest_redshift_lc['lc']['MAG'].values[(closest_redshift_lc['lc']['BAND'].values == band) & (closest_redshift_lc['lc']['NITE'].values == nite)][0] + fake_noise[noise_idx]
                        except:
                            band_df = closest_redshift_lc['lc'][closest_redshift_lc['lc']['BAND'].values == band].copy().reset_index(drop=True)
                            # set mag to 99 if nite is outside the SED
                            if nite < band_df['NITE'].values.min() or nite > band_df['NITE'].values.max():
                                mag = 99.0
                            # if nite is within bounds of SED, linearly interpolate
                            else:
                                closest_nite_indices = np.abs(nite - band_df['NITE'].values).argsort()[:2]
                                mag = (band_df['MAG'].values[closest_nite_indices[1]] - band_df['MAG'].values[closest_nite_indices[0]]) * (nite - band_df['NITE'].values[closest_nite_indices[1]]) / (band_df['NITE'].values[closest_nite_indices[1]] - band_df['NITE'].values[closest_nite_indices[0]]) + band_df['MAG'].values[closest_nite_indices[1]]
                                
                            output_dict[band][obj_string + '-magnitude' + suffix] = mag
                            output_dict[band][obj_string + '-magnitude_measured' + suffix] = np.random.normal(loc=mag, scale=0.03)
                                
                        
                    output_dict[band][obj_string + '-nite'] = orig_nite
                    output_dict[band][obj_string + '-peaknite'] = peakshift
                    output_dict[band][obj_string + '-id'] = closest_redshift_lc['sed']
                    output_dict[band][obj_string + '-type'] = closest_redshift_lc['obj_type']

                # Use independent observing conditions for each nite if conditions are drawn from distributions
                # seeing
                if isinstance(self.main_dict["SURVEY"]["PARAMETERS"]["seeing"], dict):
                    output_dict[band]["seeing"] = self._draw(self.main_dict["SURVEY"]["PARAMETERS"]["seeing"]["DISTRIBUTION"], bands=band)[0]
                # sky_brightness
                if isinstance(self.main_dict["SURVEY"]["PARAMETERS"]["sky_brightness"], dict):
                    output_dict[band]["sky_brightness"] = self._draw(self.main_dict["SURVEY"]["PARAMETERS"]["sky_brightness"]["DISTRIBUTION"], bands=band)[0]
                # magnitude_zero_point
                if isinstance(self.main_dict["SURVEY"]["PARAMETERS"]["magnitude_zero_point"], dict):
                    output_dict[band]["magnitude_zero_point"] = self._draw(self.main_dict["SURVEY"]["PARAMETERS"]["magnitude_zero_point"]["DISTRIBUTION"], bands=band)[0]

                    
            output_dicts.append(copy.deepcopy(output_dict))
            del output_dict
                    
        return output_dicts

    def generate_time_series(self, configuration, nites, objects, redshift_dicts, cosmo):
        """
        Generate a light curve bank for each configuration with timeseries info

        Args:
            configuration (str): like 'CONFIGURATION_1', 'CONFIGURATION_2', etc...
            nites (List[int] or str): a list of nites relative to explosion to get a photometric measurement or the name of a cadence file  
            objects (List[str]):  a list of object names   
            redshift_dicts (List[dict]): a list of redshift information about the objects
            cosmo (astropy.cosmology): An astropy.cosmology instance for distance calculations
        """

        # Convert nites to a cadence dict
        if isinstance(nites, str):
            cadence_dict = read_cadence_file(nites)
        else:
            cadence_dict = {'REFERENCE_MJD': 0.0,
                            'POINTING_1': {b: nites for b in self.main_dict['SURVEY']['PARAMETERS']['BANDS'].split(',')}}
        self.cadence_dict = cadence_dict

        # Use the reference MJD to shift all the nites to be relative to 0
        shifted_cadence_dict = {k: {b: [x - cadence_dict['REFERENCE_MJD'] for x in cadence_dict[k][b]] for b in self.main_dict['SURVEY']['PARAMETERS']['BANDS'].split(',')} for k in cadence_dict.keys() if k.startswith('POINTING_')}
            
        # instantiate an LCGen object
        lc_gen = timeseries.LCGen(bands=self.main_dict['SURVEY']['PARAMETERS']['BANDS'])

        # make a library for each pointing - need to speed this up (horrible performance for non-fixed redshifts and many pointings)
        for pointing, nite_dict in shifted_cadence_dict.items():
            
            for obj, redshift_dict in zip(objects, redshift_dicts):
                lc_library = []
                
                # get redshifts to simulate light curves at
                if isinstance(redshift_dict, dict):
                    drawn_redshifts = [self._draw(redshift_dict['DISTRIBUTION'], bands='g') for _ in range(100)]
                    redshifts = np.linspace(np.min(drawn_redshifts), np.max(drawn_redshifts), 15)
                else:
                    redshifts = np.array([redshift_dict])

                # get model to simulate
                model_info = self.main_dict['SPECIES'][self._species_map[obj]]['MODEL'].split('_')
                if model_info[-1].lower() == 'random' or len(model_info) == 1:
                    for redshift in redshifts:
                        lc_library.append(eval('lc_gen.gen_{0}(redshift, nite_dict, cosmo=cosmo)'.format(model_info[0])))
                else:
                    for redshift in redshifts:
                        lc_library.append(eval('lc_gen.gen_{0}(redshift, nite_dict, sed_filename="{1}", cosmo=cosmo)'.format(model_info[0], model_info[1])))
            
                setattr(self, configuration + '_' + obj + '_lightcurves_' + pointing, {'library': lc_library, 'redshifts': redshifts})
        
        return
    
    def breakup(self, verbose=False):
        """
        Based on configurations and dataset size, build list of simulation dicts.

        Args:
            verbose (bool, optional, default=False): Automatically passed from deeplenstronomy.make_dataset() args.
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

        # Set cosmology information
        cosmology_info = ['H0', 'Om0', 'Tcmb0', 'Neff', 'm_nu', 'Ob0']
        cosmo = FlatLambdaCDM(**dict_select_choose(configurations[k]['COSMOLOGY_DICT'], cosmology_info))
            
        # Add noise metadata
        for k in configurations.keys():
            noise_dict = {}
            number_of_noise_sources = len([x for x in self.main_dict['GEOMETRY'][k].keys() if x.find('NOISE_SOURCE') != -1])
            noise_dict['NUMBER_OF_NOISE_SOURCES'] = number_of_noise_sources
            for noise_source_idx in range(number_of_noise_sources):
                noise_source_num = noise_source_idx + 1
                noise_dict['NOISE_SOURCE_{0}-NAME'.format(noise_source_num)] = self.main_dict['GEOMETRY'][k]['NOISE_SOURCE_{0}'.format(noise_source_num)]
            configurations[k]['NOISE_DICT'] = noise_dict
            
        # Check for timeseries metadata
        for k in configurations.keys():
            setattr(self, k + '_time_series', False)
            if 'TIMESERIES' in self.main_dict['GEOMETRY'][k].keys():
                
                # Make a directory to store light curve data
                if not os.path.exists(self.main_dict['DATASET']['PARAMETERS']['OUTDIR']):
                    os.mkdir(self.main_dict['DATASET']['PARAMETERS']['OUTDIR'])
                    
                if not os.path.exists('{0}/lightcurves'.format(self.main_dict['DATASET']['PARAMETERS']['OUTDIR'])):
                    os.mkdir('{0}/lightcurves'.format(self.main_dict['DATASET']['PARAMETERS']['OUTDIR']))

                # Find the plane of the ojects and save the redshift sub-dict
                redshift_dicts = []
                for obj_name in self.main_dict['GEOMETRY'][k]['TIMESERIES']['OBJECTS']:
                    for sub_k in self.main_dict['GEOMETRY'][k].keys():
                        if sub_k[0:5] == 'PLANE':
                            for sub_sub_k in self.main_dict['GEOMETRY'][k][sub_k].keys():
                                if sub_sub_k[0:6] == 'OBJECT':
                                    if self.main_dict['GEOMETRY'][k][sub_k][sub_sub_k] == obj_name:
                                        if isinstance(self.main_dict['GEOMETRY'][k][sub_k]['PARAMETERS']['REDSHIFT'], dict):
                                            redshift_dicts.append(self.main_dict['GEOMETRY'][k][sub_k]['PARAMETERS']['REDSHIFT'].copy())
                                        else:
                                            redshift_dicts.append(self.main_dict['GEOMETRY'][k][sub_k]['PARAMETERS']['REDSHIFT'] + 0.0)

                if verbose: print("Generating time series data for {0}".format(k))

                # If light curves already exist, skip generation
                #if os.path.exists('{0}/lightcurves/{1}_{2}.npy'.format(self.main_dict['DATASET']['PARAMETERS']['OUTDIR'], k, self.main_dict['GEOMETRY'][k]['TIMESERIES']['OBJECTS'][0])):
                #    setattr(self, k + '_{0}_lightcurves'.format(self.main_dict['GEOMETRY'][k]['TIMESERIES']['OBJECTS'][0]), np.load('{0}/lightcurves/{1}_{2}.npy'.format(self.main_dict['DATASET']['PARAMETERS']['OUTDIR'], k, self.main_dict['GEOMETRY'][k]['TIMESERIES']['OBJECTS'][0])), allow_pickle=True) 
                #else:
                #    self.generate_time_series(k, self.main_dict['GEOMETRY'][k]['TIMESERIES']['NITES'], self.main_dict['GEOMETRY'][k]['TIMESERIES']['OBJECTS'], redshift_dicts, cosmo)
                #    np.save('{0}/lightcurves/{1}_{2}.npy'.format(self.main_dict['DATASET']['PARAMETERS']['OUTDIR'], k, self.main_dict['GEOMETRY'][k]['TIMESERIES']['OBJECTS'][0]), eval('self.' + k + '_{0}_lightcurves'.format(self.main_dict['GEOMETRY'][k]['TIMESERIES']['OBJECTS'][0])), allow_pickle=True)

                # Generate the time-series data
                self.generate_time_series(k, self.main_dict['GEOMETRY'][k]['TIMESERIES']['NITES'], self.main_dict['GEOMETRY'][k]['TIMESERIES']['OBJECTS'], redshift_dicts, cosmo)
                setattr(self, k + '_time_series', True)

                
        # For each configuration, generate full sim info for as many objects as user specified
        configuration_sim_dicts = {}
        if verbose: print("Entering main organization loop")
        for k, v in configurations.items():
            if verbose: print("Organizing {0}".format(k))
            configuration_sim_dicts[k] = []

            time_series = eval('self.{0}_time_series'.format(k))
            if time_series:
                # Get string referencing the varaible object
                obj_strings = [self._find_obj_string(x, k) for x in self.main_dict['GEOMETRY'][k]['TIMESERIES']['OBJECTS']]

                # Get the PEAK for the configuration
                if 'PEAK' in self.main_dict['GEOMETRY'][k]['TIMESERIES'].keys():
                    if isinstance(self.main_dict['GEOMETRY'][k]['TIMESERIES']['PEAK'], dict):
                        peakshifts = [self._draw(self.main_dict['GEOMETRY'][k]['TIMESERIES']['PEAK']['DISTRIBUTION'], bands='b')[0] for _ in range(v['SIZE'])]
                    else:
                        peakshifts = [float(self.main_dict['GEOMETRY'][k]['TIMESERIES']['PEAK'])] * v['SIZE']
                else:
                    peakshifts = [0.0] * v['SIZE']

            # Handle forced inputs
            inputs = {}
            for force_param, values in self.forced_inputs.items():
                configuration, param_name, band = force_param
                if configuration == k:
                    inputs[(param_name, band)] = values

            input_df = pd.DataFrame(inputs)
                    
            
            for objid in range(v['SIZE']):

                if time_series:
                    flattened_image_infos = self._flatten_and_fill_time_series(v.copy(), cosmo, k, obj_strings, objid, peakshifts[objid], inputs=input_df.loc[objid] if len(input_df) != 0 else None)
                    for flattened_image_info in flattened_image_infos:
                        configuration_sim_dicts[k].append(flattened_image_info)
                else:
                    configuration_sim_dicts[k].append(self._flatten_and_fill(v.copy(), cosmo, input_df.loc[objid] if len(input_df) != 0 else None, objid))    

        self.configuration_sim_dicts = configuration_sim_dicts



        

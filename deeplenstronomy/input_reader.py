# Classes to parse input yaml files and organize user settingsOA

from astropy.cosmology import FlatLambdaCDM
from benedict import benedict
import copy
import random
import numpy as np
import yaml

import deeplenstronomy.timeseries as timeseries
from deeplenstronomy.utils import dict_select, dict_select_choose
import deeplenstronomy.distributions as distributions
import deeplenstronomy.special as special
import deeplenstronomy.check as big_check

class Parser():
    """ 
    Load yaml inputs into a single dictionary. Check for user errors

    :param config: main yaml file name

    """

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
        big_check.run_checks(self.full_dict, self.config_dict)
        
        #BIG TODO
        
        
        #don't name an object 'None'
        #object names must be unique
        #spell 'parameters' right for once in your life
        
        #sigma_bkg versus background_rms depends on lenstronomy version
        
        #require point sources to be listed after the host

        # if timeseries, and num exposures > 1 issue a warning
        
        return
    

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

    def _choose_position(self, ra_host, dec_host, sep):
        """
        Select an ra and dec that will be sep away from the host
        
        :param ra_host: x-coord of point source host
        :param dec_host: y-coord of point source host
        :param sep: angular separation between point source and host
        :return: chosen_ra: x-coord of chosen point sep away from host
        :return: chosen_dec: y-coord of chosen point sep away from host
        """
        angle = random.uniform(0.0, 2 * np.pi)
        chosen_ra = np.cos(angle) * sep + ra_host
        chosen_dec = np.sin(angle) * sep + dec_host
        return chosen_ra, chosen_dec

    def _find_obj_string(self, obj_name, configuration):
        """
        Return the location of an object in the flattened dictionary
        
        :param obj_name: the name of the object
        :param configuration: 'CONFIGURATION_1', 'CONFIGURATION_2', etc.
        :return: obj_string: the location of the object in the flattened dictionary
        """

        d = benedict(self.main_dict['GEOMETRY'][configuration].copy(), keypath_separator='.')
        return [x.replace('.', '-') for x in d.keypaths() if d[x] == obj_name][0]

    
    def _flatten_and_fill(self, config_dict, objid=0):
        """
        Flatten input dictionary, and sample from any specified distributions
        
        :param config_dict: dictionary built up by self.breakup()
        :return: flattened_and_filled dictionary: dict ready for individual image sim
        """
        bands = config_dict['SURVEY_DICT']['BANDS'].split(',')
        output_dict = {x: {} for x in bands}

        #Object IDs
        for band in bands:
            output_dict[band]['OBJID'] = objid
        
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
                    for band, draw in zip(bands, draws):
                        for obj_num in range(1, config_dict['SIM_DICT']['PLANE_{0}-NUMBER_OF_OBJECTS'.format(plane_num)] + 1):
                            output_dict[band]['PLANE_{0}-OBJECT_{1}-{2}'.format(plane_num, obj_num, k_param)] = draw
                else:
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
                            if isinstance(self.main_dict['SPECIES'][self._species_map[obj_name]]['PARAMETERS']['sep'], dict):
                                draws = self._draw(self.main_dict['SPECIES'][self._species_map[obj_name]]['PARAMETERS']['sep']['DISTRIBUTION'], bands)
                                sep = draws[0]
                            else:
                                sep = self.main_dict['SPECIES'][self._species_map[obj_name]]['PARAMETERS']['sep']

                            ##convert image separation into ra and dec
                            ra, dec = self._choose_position(ra_host, dec_host, sep)

                        else:
                            #set ra and dec to host center
                            ra, dec = ra_host, dec_host

                        for band in bands:
                            output_dict[band]['PLANE_{0}-OBJECT_{1}-HOST'.format(plane_num, obj_num)] = self.main_dict['SPECIES'][self._species_map[obj_name]]['HOST']
                            output_dict[band]['PLANE_{0}-OBJECT_{1}-NAME'.format(plane_num, obj_num)] = obj_name
                            output_dict[band]['PLANE_{0}-OBJECT_{1}-ra'.format(plane_num, obj_num)] = ra
                            output_dict[band]['PLANE_{0}-OBJECT_{1}-dec'.format(plane_num, obj_num)] = dec
                    else:
                        #foreground - no host or sep to worry about
                        pass
                        
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
                
        return output_dict


    def _flatten_and_fill_time_series(self, config_dict, configuration, obj_strings, objid):
        """
        Generate an image info dictionary for each step in the time series

        :param config_dict: dictionary built up by self.breakup()
        :param configuration: CONFIGURATION_1, CONFIGURATION_2, etc.
        :param obj_string: list of the strings targetting the object in the flattened dictionary (e.g. ['PLANE_2-OBJECT_2'])
        :return: flattened_and_filled dictionary: dict ready for individual image sim  
        """
        output_dicts = []
        bands = self.main_dict['SURVEY']['PARAMETERS']['BANDS'].split(',')
        # Get flattened and filled dictionary
        base_output_dict = self._flatten_and_fill(config_dict, objid)

        closest_redshift_lcs = []
        for obj_name, obj_string in zip(self.main_dict['GEOMETRY'][configuration]['TIMESERIES']['OBJECTS'], obj_strings):
            # determine closest lc in library to redshift
            redshift = base_output_dict[bands[0]][obj_string + '-REDSHIFT']
            lcs = eval('self.{0}_{1}_lightcurves'.format(configuration, obj_name))
            closest_redshift_lcs.append(lcs['library'][np.argmin(np.abs(redshift - lcs['redshifts']))])

        # overwrite the image sim dictionary
        fake_noise = np.random.normal(scale=0.15, loc=0, size=len(obj_strings) * len(bands) * len(self.main_dict['GEOMETRY'][configuration]['TIMESERIES']['NITES']))
        noise_idx = 0
        for nite in self.main_dict['GEOMETRY'][configuration]['TIMESERIES']['NITES']:
            output_dict = base_output_dict.copy()
            for band in bands:
                for obj_sting, closest_redshift_lc in zip(obj_strings, closest_redshift_lcs):

                    try:
                        #try using the exact night
                        output_dict[band][obj_string + '-magnitude'] = closest_redshift_lc['lc']['MAG'].values[(closest_redshift_lc['lc']['BAND'].values == band) & (closest_redshift_lc['lc']['NITE'].values == nite)][0] + fake_noise[noise_idx]
                    except:
                        #linearly interpolate between the closest two nights
                        band_df = closest_redshift_lc['lc'][closest_redshift_lc['lc']['BAND'].values == band].copy().reset_index(drop=True)
                        closest_nite_indices = np.abs(nite - band_df['NITE'].values).argsort()[:2]
                        output_dict[band][obj_string + '-magnitude'] = (band_df['MAG'].values[closest_nite_indices[1]] - band_df['MAG'].values[closest_nite_indices[0]]) * (nite - band_df['NITE'].values[closest_nite_indices[1]]) / (band_df['NITE'].values[closest_nite_indices[1]] - band_df['NITE'].values[closest_nite_indices[0]]) + band_df['MAG'].values[closest_nite_indices[1]] + fake_noise[noise_idx]

                    output_dict[band][obj_string + '-nite'] = nite
                    output_dict[band][obj_string + '-id'] = closest_redshift_lc['sed']
                    output_dict[band][obj_string + '-type'] = closest_redshift_lc['obj_type']

                    noise_idx += 1
                    
            output_dicts.append(copy.deepcopy(output_dict))
            del output_dict
                    
        return output_dicts

    def generate_time_series(self, configuration, nites, objects, redshift_dicts, cosmo):
        """
        Generate a light curve bank for each configuration with timeseries info

        :param nites: a list of nites to get a photometric measurement
        :param objects: a list of object names
        :param redshift_dicts: a list of redshift information about the objects
        """

        # instantiate an LCGen object
        lc_gen = timeseries.LCGen(bands=self.main_dict['SURVEY']['PARAMETERS']['BANDS'])

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
            if model_info[-1].lower() == 'random':
                for redshift in redshifts:
                    lc_library.append(eval('lc_gen.gen_{0}(redshift, nites, cosmo=cosmo)'.format(model_info[0])))
            else:
                for redshift in redshifts:
                    lc_library.append(eval('lc_gen.gen_{0}(redshift, nites, sed_filename={1}, cosmo=cosmo)'.format(model_info[0], model_info[1])))
            
            setattr(self, configuration + '_' + obj + '_' + 'lightcurves', {'library': lc_library, 'redshifts': redshifts})
        
        return
    
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

        # Check for timeseries metadata
        for k in configurations.keys():
            setattr(self, k + '_time_series', False)
            if 'TIMESERIES' in self.main_dict['GEOMETRY'][k].keys():
                # Find the plane of the ojects and save the redshift sub-dict
                redshift_dicts = []
                for obj_name in self.main_dict['GEOMETRY'][k]['TIMESERIES']['OBJECTS']:
                    for sub_k in self.main_dict['GEOMETRY'][k].keys():
                        if sub_k[0:5] == 'PLANE':
                            for sub_sub_k in self.main_dict['GEOMETRY'][k][sub_k].keys():
                                if sub_sub_k[0:6] == 'OBJECT':
                                    if self.main_dict['GEOMETRY'][k][sub_k][sub_sub_k] == obj_name:
                                        redshift_dicts.append(self.main_dict['GEOMETRY'][k][sub_k]['PARAMETERS']['REDSHIFT'].copy())

                #set the cosmology for luminosity distance calculations
                cosmology_info = ['H0', 'Om0', 'Tcmb0', 'Neff', 'm_nu', 'Ob0']
                cosmo = FlatLambdaCDM(**dict_select_choose(configurations[k]['COSMOLOGY_DICT'], cosmology_info))
                                        
                self.generate_time_series(k, self.main_dict['GEOMETRY'][k]['TIMESERIES']['NITES'], self.main_dict['GEOMETRY'][k]['TIMESERIES']['OBJECTS'], redshift_dicts, cosmo)
                setattr(self, k + '_time_series', True)
            
        # For each configuration, generate full sim info for as many objects as user specified
        configuration_sim_dicts = {}
        for k, v in configurations.items():
            configuration_sim_dicts[k] = []

            time_series = eval('self.{0}_time_series'.format(k))
            if time_series:
                # Get string referencing the varaible object
                obj_strings = [self._find_obj_string(x, k) for x in self.main_dict['GEOMETRY'][k]['TIMESERIES']['OBJECTS']]

            for objid in range(v['SIZE']):

                if time_series:
                    flattened_image_infos = self._flatten_and_fill_time_series(v.copy(), k, obj_strings, objid)
                    for flattened_image_info in flattened_image_infos:
                        configuration_sim_dicts[k].append(flattened_image_info)
                else:
                    configuration_sim_dicts[k].append(self._flatten_and_fill(v.copy()))    

        self.configuration_sim_dicts = configuration_sim_dicts



        

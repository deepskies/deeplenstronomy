# Classes to parse input yaml files and organize user settings

from benedict import benedict
import numpy as np
import yaml

from image_generator import dict_select
import distributions
import special

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
        
        #BIG TODO
        
        
        #don't name an object 'None'
        #object names must be unique
        #spell 'parameters' right for once in your life
        
        #sigma_bkg versus background_rms depends on lenstronomy version
        
        #require point sources to be listed after the host
        
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
                                sep = self.main_dict['SPECIES'][self._species_map[obj_name]]['PARAMETERS']

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


        

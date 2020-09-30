# Outer shell to do everything for you

import os
import sys
import time

import h5py
import numpy as np
import pandas as pd

from deeplenstronomy.input_reader import Organizer, Parser
from deeplenstronomy.image_generator import ImageGenerator
from deeplenstronomy.utils import draw_from_user_dist, organize_image_backgrounds, read_images
from deeplenstronomy import surveys

class Dataset():
    def __init__(self, config=None, save=False, store=True):
        """
        Create a dataset. If config file or config dict is supplied, generate it.

        :param config: yaml file specifying dataset characteristics 
                       OR                                                                                                                                                                                                                                                                                                                                                                                                                                       pre-parsed y                       yaml file as a dictionary 
        :param store: If true, the generated data is stored as attributes of this object
        :param save: If true, the generated data is written to disk
        """
        
        if config:
            make_dataset(config, dataset=self, save=save, store=store)
        return

    def update_param(self, new_param_dict, configuration):
        """
        Update single parameters to new values

        :param new_param_dict: {'param_1_name': new_value_1, 'param_2_name': new_value_2, ...}
        :param configuration: like 'CONFIGURATION_1', 'CONFIGURATION_2', etc...
        """
        # Put in the updated values
        for new_param, new_value in new_param_dict.items():
            exec("self.config_dict" + self._locate(new_param, configuration) + " = new_value")
            
        return

    def update_param_dist(self, new_param_dist_dict, configuration):
        """
        Update the distribution from which a parameter is drawn

        :param new_param_dist_dict: should look like this:
            {'param_1_name': {'name': 'uniform', 
                              'parameters': {'minimum': new_value_1, 
                                             'maximum': new_value_2}},
             'param_2_name': {'name': 'uniform', 
                              'parameters': {'minimum': new_value_3, 
                                             'maximum': new_value_4}}, ...}
        :param configuration: like 'CONFIGURATION_1', 'CONFIGURATION_2', etc...   
        """
        # Put in the updated distributions
        for new_param, new_dist_info in new_param_dist_dict.items():
            location = self._locate(new_param, configuration)
            exec("self.config_dict" + location + " = {'DISTRIBUTION': {'NAME': '', 'PARAMETERS': {}}}")
            exec("self.config_dict" + location + "['DISTRIBUTION']['NAME'] = new_dist_info['name']")
            for new_dist_param, new_dist_param_value in new_dist_info['parameters'].items():
                exec("self.config_dict" + location + "['DISTRIBUTION']['PARAMETERS'][new_dist_param] = new_dist_param_value")

        return
    
    def _locate(self, param, configuration):
        """
        Find the path to the desired parameter in the main dictionary

        :param param: the name of the parameter you want to find the path for
        :param configuration: like 'CONFIGURATION_1', 'CONFIGURATION_2', etc... 
        :return: the path to the parameter
        """
        if param[-1] in self.config_dict['SURVEY']['PARAMETERS']['BANDS'].split(','):
            # Trim the band if the user left it in
            param = param[:-2]

        # DATASET- allowed params
        if param in ['SIZE', 'OUTDIR']:
            return "['DATASET']['PARAMETERS']['{0}']".format(param)
        # COSMOLOGY
        elif param in ['H0', 'Om0', 'Tcmb0', 'Neff', 'm_nu', 'Ob0']:
            return "['COSMOLOGY']['PARAMETERS']['{0}']".format(param)
        # IMAGE
        elif param in ['exposure_time', 'numPix', 'pixel_scale', 'psf_type', 'read_noise', 'ccd_gain']:
            return "['IMAGE']['PARAMETERS']['{0}']".format(param)
        # SURVEY
        elif param in ['BANDS', 'seeing', 'magnitude_zero_point', 'sky_brightness', 'num_exposures']:
            return "['SURVEY']['PARAMETERS']['{0}']".format(param)
        # NOISE
        elif param[0:5] == 'NOISE':
            # type of noise
            if param[-4:] == 'NAME':
                return "['GEOMETRY']['{0}']['{1}']".format(configuration, param.split('-')[0])
            # noise properties
            else:
                noise_name = self.config_dict['GEOMETRY'][configuration][param.split('-')[0]]
                for k, v in self.config_dict['SPECIES'].items():
                    for v_key in v.keys():
                        if v_key[0:5] == 'NOISE':
                            if v['NAME'] == noise_name:
                                return "['SPECIES']['{0}']['PARAMETERS']['{1}']".format(k, param.split('-')[-1])
            
        # GEOMETRY and SPECIES
        elif param[0:5] == 'PLANE':
            # redshift
            if param[-8:] == 'REDSHIFT':
                return "['GEOMETRY']['{0}']['{1}']['PARAMETERS']['REDSHIFT']".format(configuration, param.split('-')[0])
            # timeseries - not available yet
            
            # species
            else:
                obj_name = self.config_dict['GEOMETRY'][configuration][param.split('-')[0]][param.split('-')[1]]
                for k, v in self.config_dict['SPECIES'].items():
                    if 'NAME' in v.keys():
                        if obj_name == self.config_dict['SPECIES'][k]['NAME']:
                            return "['SPECIES']['{0}']['{1}']['PARAMETERS']['{2}']".format(k, param.split('-')[2], param.split('-')[-1])
                print("If you're seeing this message, you're trying to update something I wasn't prepared for.\nShoot me a message on Slack")
        else:
            print("If you're seeing this message, you're trying to update something I wasn't prepared for.\nShoot me a message on Slack")

        
    def regenerate(self, **make_dataset_args):
        """
        Using the dictionary stored in self.config_dict, make a new dataset
        
        :param make_dataset_args: dict, arguments supplied to make_dataset when original dataset was generated
        """
        params = dict(**make_dataset_args)
        params['config'] = self.config_dict
        params['dataset'] = self
        make_dataset(**params)
        return

def flatten_image_info(sim_dict):
    """
    Sim dict will have structure 
        {'g': {'param1': value1}, 'r': {'param1': value2} ...}
    This function will change the structure to
        {'param1_g': value1, 'param1_r': value2, ...}

    :param sim_dict: input sim_dict for ImageGenerator class
    :returns out_dict: flattened sim_dict
    """
    out_dict = {}
    for k, v in sim_dict.items():
        for sim_param, sim_value in v.items():
            out_dict[sim_param + '_' + k] = sim_value

    return out_dict
    
def get_forced_sim_inputs(forced_inputs, configurations, bands):

    force_param_inputs = {}
    for force_params in forced_inputs.values():
        for name_idx, name in enumerate(force_params['names']):
            prefices, suffices = [], []
            # Configuration dependence  
            if name.startswith("CONFIGURATION"):
                prefices = [name.split('-')[0]]
                param_name = '-'.join(name.split('-')[1:])
            else:
                prefices = configurations[:]
                param_name = name
            # Color dependence
            for b in bands:
                if name.endswith('-' + b):
                    suffices = [b]
                    param_name = '-'.join(param_name.split('-')[:-1])
                    break
            if len(suffices) == 0:
                suffices = bands[:]
                param_name = param_name # this is not necessary at all, but makes me feel good inside seeing it match with the other blocks

            # Duplicate drawn values to necessary configurations / bands
            for prefix in prefices:
                for suffix in suffices:
                    if len(np.shape(force_params['values'])) == 1:
                        force_param_inputs[(prefix, param_name, suffix)] = force_params['values']
                    else:
                        #numpy array for multiple dimensions
                        force_param_inputs[(prefix, param_name, suffix)] = force_params['values'][:,name_idx]

    return force_param_inputs

def _check_survey(survey):
    if survey is None:
        return True
    else:
        return survey in dir(surveys)   

def _format_time(elapsed_time):
    """
    Format a number of seconds as a HHMMSS string

    :param elapsed_time: float, an amount of time in seconds
    :return time_string: a formatted string of the elapsed time
    """
    hours = elapsed_time // 3600
    minutes = (elapsed_time - hours * 3600) // 60
    seconds = elapsed_time - (hours * 3600) - (minutes * 60)
    return "%i H %i M %i S" %(hours, minutes, seconds)

def make_dataset(config, dataset=None, save_to_disk=False, store_in_memory=True,
                 verbose=False, store_sample=False, image_file_format='npy',
                 survey=None, return_planes=False, skip_image_generation=False,
                 solve_lens_equation=False):
    """
    Generate a dataset from a config file

    :param config: yaml file specifying dataset characteristics
                   OR
                   pre-parsed yaml file as a dictionary
    :param verbose: if true, print status updates
    :param store_in_memory: save images and metadata as attributes
    :param save_to_disk: save images and metadata to disk
    :param store_sample: save five images and metadata as attribute
    :param image_file_format: outfile format type (npy, h5)
    :param survey: str, a default astronomical survey to use
    :param return_planes: bool, if true, return the separate planes of simulated images
    :param skip_image_generation: bool, if true, skip image generation
    :param solve_lens_equation: bool, if true, calculate the source positions
    :return: dataset: instance of dataset class
    """

    if solve_lens_equation and skip_image_generation:
        raise RuntimeError("You cannot skip image generation and solve the lens equation")
    
    if dataset is None:
        dataset = Dataset()
    else:
        parser = dataset.parser

    # set arguments of dataset generation
    dataset.arguments = dict(**locals())

    if isinstance(config, dict):
        dataset.config_dict = config
    else:    
        # Store config file
        dataset.config_file = config
        
        # Parse the config file and store config dict
        if not _check_survey(survey):
            raise RuntimeError("survey={0} is not a valid survey.".format(survey))
        parser = Parser(config, survey=survey)
        dataset.config_dict = parser.config_dict

    # Store top-level dataset info
    dataset.name = dataset.config_dict['DATASET']['NAME']
    dataset.size = dataset.config_dict['DATASET']['PARAMETERS']['SIZE']
    dataset.outdir = dataset.config_dict['DATASET']['PARAMETERS']['OUTDIR']
    dataset.bands = dataset.config_dict['SURVEY']['PARAMETERS']['BANDS'].split(',')

    # Make the output directory if it doesn't exist already
    if save_to_disk:
        if not os.path.exists(dataset.outdir):
            os.mkdir(dataset.outdir)
    
    # Organize the configuration dict
    organizer = Organizer(dataset.config_dict, verbose=verbose)

    # Store configurations
    dataset.configurations = list(organizer.configuration_sim_dicts.keys())

    # Store species map
    dataset.species_map = organizer._species_map

    # If user-specified distributions exist, draw from them
    forced_inputs = {}
    for fp in parser.file_paths:
        filename = eval("parser.config_dict['" + fp.replace('.', "']['") + "']" + "['FILENAME']")
        mode = eval("parser.config_dict['" + fp.replace('.', "']['") + "']" + "['MODE']")
        try:
            step = eval("parser.config_dict['" + fp.replace('.', "']['") + "']" + "['STEP']")
        except KeyError:
            step = 10
        draw_param_names, draw_param_values = draw_from_user_dist(filename, dataset.size, mode, step)
        forced_inputs[filename] = {'names': draw_param_names, 'values': draw_param_values}
    
    # Overwrite the configuration dict with any forced values from user distribtuions
    force_param_inputs = get_forced_sim_inputs(forced_inputs, dataset.configurations, dataset.bands)

    for force_param, values in force_param_inputs.items():
        configuration, param_name, band = force_param

        sim_inputs = organizer.configuration_sim_dicts[configuration]
        for sim_input, val in zip(sim_inputs, values):
            sim_input[band][param_name] = val

    # Skip image generation if desired
    if skip_image_generation:
        # Handle metadata and return dataset object
        for configuration, sim_inputs in organizer.configuration_sim_dicts.items():
            metadata = [flatten_image_info(image_info) for image_info in sim_inputs]
            metadata_df = pd.DataFrame(metadata)
            del metadata
            if save_to_disk:
                metadata_df.to_csv('{0}/{1}_metadata.csv'.format(dataset.outdir, configuration), index=False)
            if store_in_memory:
                setattr(dataset, '{0}_metadata'.format(configuration), metadata_df)
        return dataset
                
    # Initialize the ImageGenerator
    ImGen = ImageGenerator(return_planes, solve_lens_equation)

    # Handle image backgrounds if they exist
    if len(parser.image_paths) > 0:
        im_dir = parser.config_dict['BACKGROUNDS']
        image_backgrounds = read_images(im_dir, parser.config_dict['IMAGE']['PARAMETERS']['numPix'], dataset.bands)
    else:
        image_backgrounds = np.zeros((len(dataset.bands), parser.config_dict['IMAGE']['PARAMETERS']['numPix'], parser.config_dict['IMAGE']['PARAMETERS']['numPix']))[np.newaxis,:]
    
    # Simulate images
    for configuration, sim_inputs in organizer.configuration_sim_dicts.items():

        if verbose:
            print("Generating images for {0}".format(configuration))
            start_time = time.time()
            counter = 0
            total = len(sim_inputs)

        # Handle image backgrounds if they exist
        if len(parser.image_paths) > 0:
            image_indices = organize_image_backgrounds(im_dir, len(image_backgrounds), [flatten_image_info(sim_input) for sim_input in sim_inputs])
        else:
            image_indices = np.zeros(len(sim_inputs), dtype=int) 
            
        metadata, images = [], []
        if return_planes:
            planes = []

        for image_info, image_idx in zip(sim_inputs, image_indices):
            # track progress if verbose
            if verbose:
                counter += 1
                if counter % 50 == 0:
                    progress = counter / total * 100
                    elapsed_time = time.time() - start_time
                    sys.stdout.write('\r\tProgress: %.1f %%  ---  Elapsed Time: %s' %(progress, _format_time(elapsed_time)))
                    sys.stdout.flush()
            
            # Add background image index to image_info
            for band in dataset.bands:
                image_info[band]['BACKGROUND_IDX'] = image_idx
                
            # make the image
            simulated_image_data = ImGen.sim_image(image_info)
            if not return_planes:
                images.append(simulated_image_data['output_image'])
            else:
                images.append(simulated_image_data['output_image'])
                planes.append(np.array([simulated_image_data['output_lens_plane'],
                                        simulated_image_data['output_source_plane'],
                                        simulated_image_data['output_point_source_plane'],
                                        simulated_image_data['output_noise_plane']]))

            if solve_lens_equation:
                for band in dataset.bands:
                    image_info[band]['x_mins'] = ';'.join([str(x) for x in simulated_image_data['x_mins']])
                    image_info[band]['y_mins'] = ';'.join([str(x) for x in simulated_image_data['y_mins']])
                    image_info[band]['num_source_images'] = simulated_image_data['num_source_images']
                              
            # Save metadata for each simulated image 
            metadata.append(flatten_image_info(image_info))

            # update the progress if in verbose mode
            if verbose:
                elapsed_time = time.time() - start_time
                if counter == len(sim_inputs):
                    sys.stdout.write('\r\tProgress: 100.0 %%  ---  Elapsed Time: %s\n' %(_format_time(elapsed_time)))
                    sys.stdout.flush()

                              
        # Group images -- the array index will correspond to the id_num of the metadata
        configuration_images = np.array(images)

        # Group planes if requested
        if return_planes:
            configuration_planes = np.array(planes)

        # Add image backgrounds -- will just add zeros if no backgrounds have been specified
        configuration_images += image_backgrounds[image_indices]
        
        # Convert the metadata to a dataframe
        metadata_df = pd.DataFrame(metadata)
        del metadata

        # Save the images and metadata to the outdir if desired (ideal for large simulation production)
        if save_to_disk:
            #Images
            if image_file_format == 'npy':
                np.save('{0}/{1}_images.npy'.format(dataset.outdir, configuration), configuration_images)
                if return_planes:
                    np.save('{0}/{1}_planes.npy'.format(dataset.outdir, configuration), configuration_planes)
            elif image_file_format == 'h5':
                hf = h5py.File('{0}/{1}_images.h5'.format(dataset.outdir, configuration), 'w')
                hf.create_dataset(dataset.name, data=configuration_images)
                hf.close()
                if return_planes:
                    hf = h5py.File('{0}/{1}_planes.h5'.format(dataset.outdir, configuration), 'w')
                    hf.create_dataset(dataset.name, data=configuration_planes)
                    hf.close()
            else:
                print("ERROR: {0} is not a supported argument for image_file_format".format(image_file_format))
            #Metadata
            metadata_df.to_csv('{0}/{1}_metadata.csv'.format(dataset.outdir, configuration), index=False)

        # Store the images and metadata to the Dataset object (ideal for small scale testing)
        if store_in_memory:
            setattr(dataset, '{0}_images'.format(configuration), configuration_images)
            setattr(dataset, '{0}_metadata'.format(configuration), metadata_df)
            if return_planes:
                setattr(dataset, '{0}_planes'.format(configuration), configuration_planes)
        elif store_sample:
            setattr(dataset, '{0}_images'.format(configuration), configuration_images[0:5].copy())
            setattr(dataset, '{0}_metadata'.format(configuration), metadata_df.iloc[0:5].copy())
            del configuration_images
            del metadata_df                
            if return_planes:
                setattr(dataset, '{0}_planes'.format(configuration), configuration_planes[0:5].copy())
                del configuration_planes
        else:
            # Clean up things that are done to save space
            del configuration_images
            del metadata_df
            if return_planes:
                del configuration_planes

    # store parser and organizer as attributes
    dataset.parser = parser
    #dataset.organizer = organizer
                
    return dataset

    
if __name__ == "__main__":
    pass

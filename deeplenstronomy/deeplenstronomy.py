# Outer shell to do everything for you

from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from deeplenstronomy.input_reader import Organizer, Parser
from deeplenstronomy.image_generator import ImageGenerator

class Dataset():
    def __init__(self, config=None, save=False, store=True):
        """
        Create a dataset. If config file or config dict is supplied, generate it.

        :param config: yaml file specifying dataset characteristics 
                       OR                                                                                                                                                                                                                                                                                                                                                                                                                                       pre-parsed yaml file as a dictionary 
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

        # COSMOLOGY
        if param in ['H0', 'Om0', 'Tcmb0', 'Neff', 'm_nu', 'Ob0']:
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

        
    def regenerate(self, save=False, store=True):
        """
        Using the dictionary stored in self.config_dict, make a new dataset
        
        :param store: If true, the generated data is stored as attributes of this object
        :param save: If true, the generated data is written to disk
        """
        make_dataset(config=self.config_dict, dataset=self, save=save, store=store)
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
    

def make_dataset(config, dataset=None, save=False, store=True):
    """
    Generate a dataset from a config file

    :param config: yaml file specifying dataset characteristics
                   OR
                   pre-parsed yaml file as a dictionary
    :return: dataset: instance of dataset class
    """

    if not dataset:
        dataset = Dataset()

    if isinstance(config, dict):
        dataset.config_dict = config
    else:    
        # Store config file
        dataset.config_file = config
        
        # Parse the config file and store config dict
        P = Parser(config)
        dataset.config_dict = P.config_dict

    # Store top-level dataset info
    dataset.name = dataset.config_dict['DATASET']['NAME']
    dataset.size = dataset.config_dict['DATASET']['PARAMETERS']['SIZE']
    dataset.outdir = dataset.config_dict['DATASET']['PARAMETERS']['OUTDIR']

    # Make the output directory if it doesn't exist already
    if save:
        if not os.path.exists(dataset.outdir):
            os.mkdir(dataset.outdir)
    
    # Organize the configuration dict
    O = Organizer(dataset.config_dict)

    # Store species map
    dataset.species_map = O._species_map

    # Store configurations
    dataset.configurations = list(O.configuration_sim_dicts.keys())

    # Initialize the ImageGenerator
    ImGen = ImageGenerator()
    
    # Simulate images
    for configuration, sim_inputs in O.configuration_sim_dicts.items():

        #print(configuration)
        
        metadata, images = [], []

        for image_info in sim_inputs:
            # Save metadata for each simulated image 
            metadata.append(flatten_image_info(image_info))

            # make the image
            images.append(ImGen.sim_image(image_info))

        # Group images -- the array index will correspond to the id_num of the metadata
        configuration_images = np.array(images)

        # Convert the metadata to a dataframe
        metadata_df = pd.DataFrame(metadata)
        del metadata

        # Save the images and metadata to the outdir if desired (ideal for large simulation production)
        if save:
            np.save('{0}/{1}_images.npy'.format(dataset.outdir, configuration), configuration_images)
            metadata_df.to_csv('{0}/{1}_metadata.csv'.format(dataset.outdir, configuration))

        # Store the images and metadata to the Dataset object (ideal for small scale testing)
        if store:
            setattr(dataset, '{0}_images'.format(configuration), configuration_images)
            setattr(dataset, '{0}_metadata'.format(configuration), metadata_df)
                                           
        else:
            # Clean up things that are done to save space
            del configuration_images
            del metadata_df
    
    return dataset


def view_image(image):
    if len(np.shape(image)) > 2:
        #multi-band mode
        fig, axs = plt.subplots(1, np.shape(image)[0])
        for index, single_band_image in enumerate(image):
            axs[index].matshow(np.log10(single_band_image))
            axs[index].set_xticks([], [])
            axs[index].set_yticks([], [])

        fig.tight_layout()
        plt.show(block=True)
        plt.close()

    else:
        #simgle-band mode
        plt.figure()
        plt.matshow(np.log10(image))
        plt.xticks([], [])
        plt.yticks([], [])
        plt.show(block=True)
        plt.close()

def view_image_rgb(image, Q=2.0, stretch=4.0):

    assert len(image) > 2
    
    rgb = make_lupton_rgb(np.log10(image[2]),
                          np.log10(image[1]),
                          np.log10(image[0]),
                          Q=Q, stretch=stretch)

    plt.figure()
    plt.imshow(rgb)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.show(block=True)
    plt.close()

    
if __name__ == "__main__":
    #make_dataset('configs/fake_config.yaml', store=False, save=False)
    pass

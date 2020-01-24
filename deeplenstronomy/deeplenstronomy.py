# Outer shell to do everything for you

from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from deeplenstronomy.input_reader import Organizer, Parser
from deeplenstronomy.image_generator import ImageGenerator

class Dataset():
    def __init__(self, config_file=None, save=False, store=True):
        if config_file:
            make_dataset(config_file, dataset=self, save=save, store=store)
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
    

def make_dataset(config_file, dataset=None, save=False, store=True):
    """
    Generate a dataset from a config file

    :param config_file: yaml file specifying dataset characteristics
    :return: dataset: instance of dataset class
    """

    if not dataset:
        dataset = Dataset()

    # Store config file
    dataset.config_file = config_file

    # Parse the config file and store config dict
    P = Parser(config_file)
    dataset.config_dict = P.config_dict

    # Store top-level dataset info
    dataset.name = P.config_dict['DATASET']['NAME']
    dataset.size = P.config_dict['DATASET']['PARAMETERS']['SIZE']
    dataset.outdir = P.config_dict['DATASET']['PARAMETERS']['OUTDIR']

    # Make the output directory if it doesn't exist already
    if save:
        if not os.path.exists(dataset.outdir):
            os.mkdir(dataset.outdir)
    
    # Organize the configuration dict
    O = Organizer(P.config_dict)

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
    make_dataset('configs/fake_config.yaml', store=False, save=False)

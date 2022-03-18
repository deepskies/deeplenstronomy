import unittest2

import sys
Folder='../deeplenstronomy'
sys.path.append(Folder)

import os
import numpy as np

import deeplenstronomy.deeplenstronomy as dl
from deeplenstronomy.input_reader import Parser

class test_make_dataset(unittest2.TestCase):

    @classmethod
    def setUpClass(self):
        self.filename= '../Notebooks/data/demo.yaml'
        self.config_dict = Parser(self.filename, survey=None).config_dict

    def test_make_from_dict(self):
        dataset_from_file = dl.make_dataset(self.filename, random_seed=42)
        images_from_file = dataset_from_file.CONFIGURATION_1_images

        dataset_from_dict = dl.make_dataset(self.config_dict, random_seed=42)
        images_from_dict = dataset_from_dict.CONFIGURATION_1_images

        # Test that datasets generated from dict and from .yaml file with this dict are the same
        self.assertTrue((images_from_file==images_from_dict).all())

    def test_no_save_load_of_sim_dicts(self):
        dataset = dl.make_dataset(self.config_dict, save_to_disk=False, random_seed=42)
        # create temporary directory
        os.system('mkdir temp')
        # replicate previous behaviour
        np.save("temp/{0}_sim_dicts.npy".format( dataset.configurations[0]),
                {0: dataset.organizer.configuration_sim_dicts[dataset.configurations[0]]}, allow_pickle=True)
        saved_loaded_file = np.load("temp/{1}_sim_dicts.npy".format(dataset.outdir, dataset.configurations[0]),
                    allow_pickle=True).item()[0]
        # remove temporary directory
        os.system('rm -r temp')

        # results of new behaviour
        original_file=dataset.organizer.configuration_sim_dicts[dataset.configurations[0]]

        similarity_arr = []
        for i, band_dict in enumerate(saved_loaded_file):
            for band in band_dict.keys():
                for key in band_dict[band].keys():
                    similarity_arr += [saved_loaded_file[i][band][key] == original_file[i][band][key]]

        # Check that I use the same as I've previously received from save-load
        self.assertTrue(np.array(similarity_arr).all())

        # Test that no files are saved if save_to_disk=False
        self.assertTrue(not np.isin('MySimulationResults',os.listdir('./')).item())

        # Check that data generated from new realisation is correct
        images_from_dict = dataset.CONFIGURATION_1_images
        self.assertTrue(images_from_dict.shape==(24,5,100,100))


if __name__ == '__main__':
    unittest2.main()

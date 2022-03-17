import unittest2

import sys
Folder='../deeplenstronomy'
sys.path.append(Folder)

import os

import deeplenstronomy.deeplenstronomy as dl
from deeplenstronomy.input_reader import Parser

class test_make_dataset_from_dict(unittest2.TestCase):

    def test(self):
        filename = '../Notebooks/data/demo.yaml'

        dataset_from_file = dl.make_dataset(filename, random_seed=42)
        images_from_file = dataset_from_file.CONFIGURATION_1_images

        config_dict = Parser(filename, survey=None).config_dict
        dataset_from_dict = dl.make_dataset(config_dict, random_seed=42)
        images_from_dict = dataset_from_dict.CONFIGURATION_1_images

        os.system('rm -r ./MySimulationResults')
        self.assertTrue((images_from_file==images_from_dict).all())

if __name__ == '__main__':
    unittest2.main()

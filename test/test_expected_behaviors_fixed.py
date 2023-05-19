"""
Parsed Config File Produces Expected Behaviors - fixed parameters
"""
import inspect
import os
import sys
#sys.path.insert(1, '/Users/jarugula/Research/Deeplenstronomy_issues')
import deeplenstronomy.deeplenstronomy as dl

doc = """



\tRunning tests from test_expected_behaviors_fixed.py


\tThe tests included in this module demonstrate that the values of fixed parameters
\tin the main configuration file are accurately utilized in the simulation and 
\tappear as expected in the simulation metadata. The functions are:

\t\t- test_dataset_section
\t\t\tTesting that NAME, OUTDIR, and SEED properties from the DATASET section of
\t\t\tthe main config file were properly interpretted and utilized as properties
\t\t\tof the generated dataset

\t\t- test_cosmology_section
\t\t\tTesting that the cosmological parameters from the COSMOLOGY section appear
\t\t\tas expected in the simulation metadata

\t\t- test_image_size
\t\t\tTesting that the IMAGE.numPix keyword produced simulated images with the
\t\t\texpected size.

\t\t- test_bands
\t\t\tTesting that the BANDS argument was interpretted properly and produced an
\t\t\tarray of simulated images with the expected number of bands

"""
print(doc)




# Below are all of the possible operation modes

kwargs_sets = {0: {}, # default arguments
               1: {'save_to_disk': True},
               2: {'save_to_disk': True, 'image_file_format': 'h5'},
               3: {'save_to_disk': True, 'skip_image_generation': True},
               4: {'store_in_memory': False},
               5: {'store_sample': True},
               6: {'skip_image_generation': True, 'survey': 'des'},
               7: {'solve_lens_equation': True},
               8: {'return_planes': True}
}

f = open('status.txt', 'r')
current_test = int(f.read().strip())
f.close()

# Generate the dataset
kwargs_set = kwargs_sets[current_test]
config_filename = 'config.yaml'
dataset = dl.make_dataset(config_filename, **kwargs_set)

has_images = [hasattr(dataset, x + '_images') for x in dataset.configurations]
has_metadata = [hasattr(dataset, x + '_metadata')
                for x in dataset.configurations]
has_planes = [hasattr(dataset, x + '_planes') for x in dataset.configurations]

images_exist = [os.path.exists(dataset.outdir +'/' + x + '_images.' +
                               dataset.arguments['image_file_format'])
                for x in dataset.configurations]
metadata_exist = [os.path.exists(dataset.outdir +'/' + x + '_metadata.csv')
                  for x in dataset.configurations]
planes_exist = [os.path.exists(dataset.outdir +'/' + x + '_planes.' +
                               dataset.arguments['image_file_format'])
                for x in dataset.configurations]


# Begin test functions

def test_dataset_section():
    section = dataset.config_dict['DATASET']['PARAMETERS']
    assert dataset.size == section['SIZE']
    assert dataset.outdir == section['OUTDIR']
    if 'SEED' in section.keys():
        assert dataset.seed == section['SEED']

def test_cosmology_section():
    if all(has_metadata):
        section = dataset.config_dict['COSMOLOGY']['PARAMETERS']
        for conf in dataset.configurations:
            for band in dataset.bands:
                for param, value in section.items():
                    md = eval(f'dataset.{conf}_metadata["{param}-{band}"]') 
                    assert all(md.values == value)

def test_image_size():
    if all(has_images):
        for conf in dataset.configurations:
            x = eval(f'dataset.{conf}_images').shape[-2]
            y = eval(f'dataset.{conf}_images').shape[-1]
            assert dataset.config_dict['IMAGE']['PARAMETERS']['numPix'] == x
            assert dataset.config_dict['IMAGE']['PARAMETERS']['numPix'] == y
            
def test_bands():
    config_bands = dataset.config_dict['SURVEY']['PARAMETERS']['BANDS'].split(',')
    assert config_bands == dataset.bands

    if all(has_images):
        for conf in dataset.configurations:
            b = eval(f'dataset.{conf}_images').shape[-3]
            assert len(config_bands) == b

    if all(has_metadata):
        get_band = lambda col: col.split('-')[-1]
        for conf in dataset.configurations:
            md = eval(f'dataset.{conf}_metadata').columns
            assert all([band in config_bands for band in [get_band(c) for c in md]])

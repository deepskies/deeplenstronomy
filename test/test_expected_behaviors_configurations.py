"""
Parsed Config File Produces Expected Behaviors - configurations
"""

import inspect
import os
import sys
# sys.path.insert(1, '/Users/jarugula/Research/Deeplenstronomy_issues')
sys.path.insert(1, '../deeplenstronomy')
import deeplenstronomy as dl
#import deeplenstronomy.deeplenstronomy as dl
#import pytest


doc = """



\tRunning tests from test_expected_behaviors_configurations.py


\tThe tests included in this module demonstrate that the properties of each 
\tconfiguration were simulated as expected. These properties include the 
\texpected size of each configuration, the objects and planes included, and
\twhether time-series functionalities appear as expected. The functions are:
\t\t- test_configuration_existence
\t\t\tTesting that all configurations present in the config file are found by 
\t\t\tdeeplenstronomy and are present in the simulation outputs

\t\t- test_configuration_fractions
\t\t\tTesting that the FRACTION keyword for each configuration resulted in
\t\t\tthe expected number of images for that configuration being produced

\t\t- test_timeseries
\t\t\tTime-series functionalities, if present, get tested by the function
\t\t\ttest_configuration_fractions

\t\t- test_planes_and_objects
\t\t\tTesting that each specified object and plane is was included in the 
\t\t\tsimulation and is present in the metadata corresponding to its
\t\t\tconfiguration
"""
print(doc)




# Below are all of the possible operation modes

kwargs_sets = {0: {}, # default arguments
               1: {'save_to_disk': True},
               2: {'save_to_disk': True, 'image_file_format': 'h5'},
               3: {'save_to_disk': True, 'skip_image_generation': False},
               4: {'store_in_memory': False},
               5: {'store_sample': True},
               6: {'skip_image_generation': False, 'survey': 'des'},
               7: {'solve_lens_equation': True},
               8: {'return_planes': True}
               }

f = open('status.txt', 'r')
current_test = int(f.read().strip())
f.close()
print('current test: ',current_test)


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


def test_configuration_existence():
    for conf in dataset.configurations:
        assert conf in dataset.config_dict['GEOMETRY'].keys()

def test_configuration_fractions():
    for conf in dataset.configurations:
        frac = dataset.config_dict['GEOMETRY'][conf]['FRACTION']
        simulated_images = int(frac * dataset.size)
        
        if all(has_images):
            assert eval(f'dataset.{conf}_images').shape[0] == simulated_images

        if all(has_metadata):
        #if all(metadata_exist):
            # not time-series
            if 'TIMESERIES' not in dataset.config_dict['GEOMETRY'][conf].keys():
                assert len(eval(f'dataset.{conf}_metadata')) == simulated_images

            # time-series
            else:
                nites = dataset.config_dict['GEOMETRY'][conf]['TIMESERIES']['NITES']
                md_rows = len(nites) * simulated_images
                assert md_rows == len(eval(f'dataset.{conf}_metadata'))

def test_timeseries():
    # already tested in test_configuration_fractions()
    pass

def test_planes_and_objects():
    for conf in dataset.configurations:

        if all(has_metadata):
            md = eval(f'dataset.{conf}_metadata')
        else:
            # this test requires metadata
            return
        
        number_of_planes = 0
        for plane in dataset.config_dict['GEOMETRY'][conf].keys():

            if plane.startswith('PLANE_'):
                number_of_planes += 1
                number_of_objects = 0

                for obj in dataset.config_dict['GEOMETRY'][conf][plane].keys():

                    if obj.startswith('OBJECT_'):
                        number_of_objects += 1

                        if all(has_metadata):
                            for band in dataset.bands:
                                num_md_cols = 0
                                for col in md.columns:
                                    if (col.startswith(f'{plane}-{obj}') and
                                        col.endswith(band)):
                                        num_md_cols += 1

                                # Plane and obj info in metadata for band
                                assert num_md_cols > 0

                # expected number of objects in plane
                for band in dataset.bands:
                    md_objects = md[plane + '-NUMBER_OF_OBJECTS-' + band].values 
                    assert all(md_objects == number_of_objects)

        # expected number of planes in configuration
        for band in dataset.bands:
            md_planes = md['NUMBER_OF_PLANES-' + band].values
            assert all(md_planes == number_of_planes)



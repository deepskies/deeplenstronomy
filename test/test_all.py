# Tests of deeplenstronomy

import os
os.system('rm -rf TestResults') # fresh start

import deeplenstronomy.deeplenstronomy as dl

"""
Make the dataset from a config file that utilizes all of deeplenstronomy's 
features. The  dataset will be a global variable accessible by all tests.
"""

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

# Run all tests by writing last test to a file
try:
    f = open('status.txt', 'r')
    previous_test = int(f.read().strip())
    f.close()
except FileNotFoundError:
    previous_test = -1
    
next_test = previous_test + 1
next_test = next_test if next_test in kwargs_sets.keys() else 0

f = open('status.txt', 'w+')
f.write(str(next_test))
f.close()

# Overwirte the file-based "next_test" with the following line
#next_test = 7

# Generate the dataset
kwargs_set = kwargs_sets[next_test]
config_filename = 'config.yaml'
dataset = dl.make_dataset(config_filename, **kwargs_set)


# Begin test functions

"""
Dataset Existence and Attributes
"""

def test_correct_return_class():
    assert isinstance(dataset, dl.Dataset)

def test_has_top_level_attributes():
    top_level_attributes = ['arguments',
                            'bands',
                            'config_dict',
                            'config_file',
                            'configurations',
                            'name',
                            'outdir',
                            'parser',
                            'seed',
                            'size',
                            'species_map']
    assert all([hasattr(dataset, x) for x in top_level_attributes])

"""
Arguments Produce Expected Outputs
"""

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

def test_store_in_memory():
    if dataset.arguments['store_in_memory']:
        assert all(has_metadata)
        if not dataset.arguments['skip_image_generation']:
            assert all(has_images)
        else:
            assert not any(has_images)
    else:
        assert not any(has_images) and not any(has_metadata)

def test_save_to_disk():
    if dataset.arguments['save_to_disk']:
        assert os.path.exists(dataset.outdir)
        assert all(metadata_exist)
        if not dataset.arguments['skip_image_generation']:
            assert all(images_exist)
        else:
            assert not any(images_exist)
    else:
        assert not os.path.exists(dataset.outdir)
        assert not any(metadata_exist)
        assert not any(images_exist)

def test_store_sample():
    if dataset.arguments['store_sample']:
        if not dataset.arguments['skip_image_generation']:
            assert all(has_images)
            assert all([len(eval("dataset." + x + '_images')) <= 5 for x in
                        dataset.configurations])
            if dataset.arguments['return_planes']:
                assert all([len(eval("dataset." + x + '_planes')) <= 5 for x in
                            dataset.configurations])
        assert all(has_metadata)
        assert all([len(eval("dataset." + x + '_metadata')) <= 5 for x in
                    dataset.configurations])

def test_return_planes():
    if dataset.arguments['return_planes']:
        if dataset.arguments['store_in_memory']:
            assert all(has_planes)
        if dataset.arguments['save_to_disk']:
            assert all(planes_exist)
    else:
        assert not any(has_planes)
        assert not any(planes_exist)


def test_survey():
    if dataset.arguments['survey'] is not None:
        config_path = dataset.config_file.split('/')
        if len(config_path) == 1:
            survey_file = dataset.arguments['survey'] + dataset.config_file
        else:
            survey_file = ('/'.join(config_path[0:-1]) + '/' +
                           dataset.arguments['survey'] + '_' + config_path[-1])
            assert os.path.exists(survey_file)

def test_solve_lens_equation():
    if dataset.arguments['solve_lens_equation']:
        assert not dataset.arguments['skip_image_generation']

        if (dataset.arguments['store_in_memory'] or
            dataset.arguments['store_sample']):
            for configuration in dataset.configurations:
                md = eval('dataset.' + configuration + '_metadata')
                for band in dataset.bands:
                    assert 'x_mins-' + band in md.columns
                    assert 'y_mins-' + band in md.columns
                    assert 'num_source_images-' + band in md.columns 

"""
Parsed Config File Produces Expected Behaviors - fixed parameters
"""

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

"""
Parsed Config File Produces Expected Behaviors - configurations
"""

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
            # not time-series
            if 'TIMESERIES' not in dataset.config_dict['GEOMETRY'][conf].keys():
                assert len(eval(f'dataset.{conf}_metadata')) == simulated_images

            # time-series
            else:
                nites = dataset.config_dict['GEOMETRY'][conf]['TIMESERIES']['NITES']
                md_rows = len(nites) * simulated_images
                assert md_rows == len(eval(f'dataset.{conf}_metadata'))

def test_timeseries():
    for conf in dataset.configurations:
        if 'TIMESERIES' in dataset.config_dict['GEOMETRY'][conf].keys():
            if all(has_images):
                assert len(eval(f'dataset.{conf}_images').shape) == 5

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
                    

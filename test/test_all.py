# Tests of deeplenstronomy

import os
os.system('rm -rf TestResults') # fresh start

import deeplenstronomy.deeplenstronomy as dl

"""
Make the dataset from a config file that utilizes all of deeplenstronomy's 
features. The  dataset will be a global variable accessible by all tests.
"""

# Below are all of the possible operation modes

kwargs_set_0 = {} # default arguments

kwargs_set_1 = {'save_to_disk': True}

kwargs_set_2 = {'save_to_disk': True,
                'image_file_format': 'h5'}

kwargs_set_3 = {'save_to_disk': True,
                'skip_image_generation': True}

# Edit the line below to select the kwyword arguments to use
kwargs_set = kwargs_set_0

# Generate the dataset
config_filename = 'config.yaml'
dataset = dl.make_dataset(config_filename, **kwargs_set)


# Begin test functions

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

def test_arguments_produced_expected_outputs():
    # store in memory
    has_images = [hasattr(dataset, x + '_images')
                  for x in dataset.configurations]
    has_metadata = [hasattr(dataset, x + '_metadata')
                    for x in dataset.configurations]
    if dataset.arguments['store_in_memory']:
        assert all(has_metadata)
        if not dataset.arguments['skip_image_generation']:
            assert all(has_images)
        else:
            assert not any(has_images)
    else:
        assert not any(has_attributes)

    # save to disk
    images_exist = [os.path.exists(dataset.outdir +'/' + x + '_images.' +
                                   dataset.arguments['image_file_format'])
                    for x in dataset.configurations]
    metadata_exist = [os.path.exists(dataset.outdir +'/' + x + '_metadata.csv')
                      for x in dataset.configurations]
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

    # store sample
        
def test_parser_to_dataset_map():
    # configurations matches parser
    # bands matches parser
    # outdir matches parser
    # 

    pass

def test_input_to_parser_map():
    pass

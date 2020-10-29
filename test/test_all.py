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
               4: {'store_in_memory': False}
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


# Generate the dataset
kwargs_set = kwargs_sets[next_test]
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

"""
Arguments produce expected outputs
"""

has_images = [hasattr(dataset, x + '_images') for x in dataset.configurations]
has_metadata = [hasattr(dataset, x + '_metadata')
                for x in dataset.configurations]
images_exist = [os.path.exists(dataset.outdir +'/' + x + '_images.' +
                               dataset.arguments['image_file_format'])
                for x in dataset.configurations]
metadata_exist = [os.path.exists(dataset.outdir +'/' + x + '_metadata.csv')
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
            pass
        
def test_parser_to_dataset_map():
    # configurations matches parser
    # bands matches parser
    # outdir matches parser
    # 

    pass

def test_input_to_parser_map():
    pass

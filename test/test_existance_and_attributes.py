import inspect
import os

import deeplenstronomy.deeplenstronomy as dl

doc = """



\tRunning tests from test_existance_and_attributes.py


\tThe tests included in this module demonstrate that the returned object from 
\tdeeplenstronomt.make_dataset() is the correct format. The functions are:

\t\t- test_correct_return_class
\t\t\tTesting that the output of deeplenstronomy.make_dataset() is an instance of 
\t\t\tthe Dataset class

\t\t- test_has_top_level_attributes
\t\t\tTesting that the dataset produced by deeplenstronomy.make_dataset() has 
\t\t\tall expected attributes (arguments, bands, config_dict, config_file,
\t\t\tconfigurations, name, outdir, parser, seed, size, species_map)
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

f = open('status.txt', 'r')
current_test = int(f.read().strip())
f.close()

# Generate the dataset
kwargs_set = kwargs_sets[current_test]
config_filename = 'config.yaml'
dataset = dl.make_dataset(config_filename, **kwargs_set)

# Display current arguments
print(f"\nCurrently testing the {config_filename} with the following arguments:")
print("deeplenstronomy.make_dataset(")
for arg in dataset.arguments:
    if arg != "dataset":
        print("\t", arg, ":", dataset.arguments[arg])
    else:
        print("\t", arg, ":", "None")
print(")")

print(doc)


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

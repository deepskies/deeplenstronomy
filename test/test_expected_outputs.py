"""
Arguments Produce Expected Outputs
"""
import inspect
import os
import sys
#sys.path.insert(1, '/Users/jarugula/Research/Deeplenstronomy_issues')
import deeplenstronomy.deeplenstronomy as dl


doc = """



\tRunning tests from test_expected_outputs.py


\tThe tests included in this module demonstrate that the outputs of the simulation
\tare properly save to disk or stored in memory, in accordance with the arguments
\tsupplied to deeplenstronomy.make_dataset(). The functions are:

\t\t- test_store_in_memory
\t\t\tTesting that images and metadata for all configurations get stored as 
\t\t\tattributes of the generated dataset

\t\t- test_save_to_disk
\t\t\tTesting that images and metadata for all configurations get saved to
\t\t\tdisk in the expected output file format

\t\t- test_score_sample
\t\t\tTesting that five or fewer images, planes, and metadata rows are kept if 
\t\t\tstore_sample is utilized

\t\t- test_return_planes
\t\t\tTesting that separate image planes are returned, and that they are either
\t\t\tsaved to disk or stored in memory depending on the arguments of 
\t\t\tdeeplenstronomy.make_dataset()

\t\t- test_survey
\t\t\tTesting if use of a pre-defined survey produced an new configuration file 
\t\t\tautomatically and that the survey-specific configuration file is being 
\t\t\tutilized by deeplenstronomy

\t\t- test_solve_lens_equation
\t\t\tTesting if the x_mins, y_mins, and num_sources keywords are present in the
\t\t\tmetadata, if solve_lens_equaiton is specified in the arguments of
\t\t\tdeeplenstronomy.make_dataset()

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

# Begin test functions


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

print("\nEvaluating generated data: (M = stored in memory, D = saved to disk, B = both, N = neither)")
case = {(True, True): 'B', (True, False): 'M', (False, True): 'D', (False, False): 'N'}

print("Configuration\t\t| Images\t| Metadata\t| Planes\t|")
for i in range(len(dataset.configurations)):
    print(dataset.configurations[i], '\t|',
          case[has_images[i],images_exist[i]], '\t\t|',
          case[has_metadata[i],metadata_exist[i]], '\t\t|',
          case[has_planes[i], planes_exist[i]], '\t\t|')

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

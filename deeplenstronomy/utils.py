# Helper functions and classes

import os
import sys

from astropy.io import fits
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, interp1d

def dict_select(input_dict, keys):
    """
    Trim a dictionary down to selected keys
    
    :param input_dict: full dictionary
    :param keys: list of keys desired in trimmed dict
    :return: dict: trimmed dictionary
    """
    return {k: input_dict[k] for k in keys}

def dict_select_choose(input_dict, keys):
    """
    Trim a dictionary down to selected keys, if they are in the dictionary
    
    :param input_dict: full dictionary
    :param keys: list of keys desired in trimmed dict
    :return: dict: trimmed dictionary
    """
    return {k: input_dict[k] for k in keys if k in input_dict.keys()}

def select_params(input_dict, profile_prefix):
    """
    Get just the parameters and values for a given profile prefix
    
    :param input_dict: full dictionary
    :param profile_prefix: i.e. "PLANE_1-OBJECT_2-LIGHT_PROFILE_1-"
    :return: dict: parameter dictionary for profile
    """
    params = [k for k in input_dict.keys() if k[0:len(profile_prefix)] == profile_prefix]
    return {x.split('-')[-1]: input_dict[x] for x in params if x[-4:] != 'NAME'}


class KeyPathDict(dict):
    """
    A Subclass of <dict> to enable keypath functionality. Original code is from the 
    python-benedict module https://github.com/fabiocaccamo/python-benedict 
    [Copyright (c) 2019 Fabio Caccamo, under the MIT license].
    """
    def __init__(self, base_dict, keypath_separator='.'):
        """
        Initialize a KeyPathDict by supplying the underlying dict to which 
        adding keypath functionality is desired.

        :param base_dict: dict, the dictionary to add keypaths to
        :param keypath_separator: str, the character to use to separate keys
        """
        # Inherit attributes of the base dict
        super().__init__(base_dict)

        # Set the keypath sepatator and find all nested keys
        self.keypath_separator = keypath_separator
        self.kls = self._keylists(base_dict)

        return

    def _get_keylist(self, item, parent_keys):
        """
        Recursively search for all nested dictionary keys.

        :param item: parent dictionary or value in a dictionary
        :param parent_keys: the keys of the dictionary one level up
        :return: keylist: list, list of all keys on a single level in the dictionary
        """
        keylist = []
        for key, value in item.items():
            # Collect the keys of the dictionary
            keys = parent_keys + [key]
            keylist += [keys]
            # If the value is a dict, recursively search that dict
            if isinstance(value, dict):
                keylist += self._get_keylist(value, keys)
        return keylist

    def _keylists(self, d):
        """
        Shell function to call the recursive key search

        :param d: dict, the dictionary to search
        :return: keylist: list, nested list of all keys in the dictionary
        """
        return self._get_keylist(d, [])
    
    def keypaths(self):
        """
        Join the keylists using the keypath_separator.

        :return: kps: list, all keypaths in the dictionary as strings
        """
        kps = [self.keypath_separator.join(['{}'.format(key) for key in kl]) for kl in self.kls]
        kps.sort()
        return kps

def read_distribution_file(filename):
    """
    Load the file information into a dataframe
    
    :param filename: str, the file containing the distribution     
    :return: df: pandas.DataFrame containing the tabular distribtution
    """
    df = pd.read_csv(filename, delim_whitespace=True)

    assert 'WEIGHT' in df.columns, "'WEIGHT' must be a column in {}".format(filename)

    return df
        

def draw_from_user_dist(filename, size, mode, step=10):
    """
    Interpolate a user-specified N-dimensional probability distribution and
    sample from it.

    :param filename: str, the file containing the distribution
    :param size: int, the number of times to sample the probability distribution
    :param mode: str, choose from ['interpolate', 'sample']
    :param step: int, the number of steps on the interpolation grid
    :return: parameters: list, the names of the paramters
    :return: choices: array with entries as arrays of drawn parameters
    """

    df = read_distribution_file(filename)

    parameters = [x for x in df.columns if x != 'WEIGHT']
    points = df[parameters].values
    weights = df['WEIGHT'].values

    if mode == 'interpolate':
        # 2+ Dimension case
        if len(parameters) > 1:
            # Interpolate the distribution and evaluate it on a grid of all possible parameter combinations
            interpolator = LinearNDInterpolator(points, weights, fill_value=0.0)
            grid_vectors = [np.linspace(df[x].values.min(), df[x].values.max(), step) for x in parameters]
            param_grids = np.array(np.meshgrid(*grid_vectors)).T.reshape(step**len(parameters), len(parameters))
            weighted_params = interpolator(param_grids)
    
            # Draw from the grid based on its weight
            draws = np.random.choice(np.arange(len(param_grids)), size=size, p=weighted_params/weighted_params.sum())
            choices = param_grids[draws]

        elif len(parameters) == 1:
            # Interpolate the 1D grid
            grid = np.linspace(df[parameters].values.min(), df[parameters].values.max(), step)
            interpolator = interp1d(points.flatten(), weights, fill_value=0.0)
            weighted_params = interpolator(grid)
            
            # Draw from the grid based on its weight
            choices = np.random.choice(grid, size=size, p=weighted_params/weighted_params.sum())

    elif mode == 'sample':
        index_arr = np.random.choice(np.arange(len(points), dtype=int), size=size, p=weights / weights.sum())
        choices = points[index_arr]

    else:
        raise NotImplementedError("unexpected mode passed, must be 'sample' or 'interpolate'")
            
    return parameters, choices

def read_images(im_dir, im_size, bands):
    """
    Read images into memory and resize to match simulations

    :param im_dir: path to directory of images 
    :param im_size: numPix along onle side of an image
    :param bands: list of bands used in simulation
    :return: im_array: processed images
    """
    # Load images into an array
    im_array = []
    for band in bands:
        if not os.path.exists(im_dir + '/' + band + '.fits'):
            print(im_dir + " is missing " + band + ".fits")
            sys.exit()
            
        hdu = fits.open(im_dir + '/' + band + '.fits')
        im_array.append(hdu[0].data)
        hdu.close()

    im_array = np.swapaxes(np.array(im_array), 0, 1)
    
    # Resize the images to match the simulations
    if im_array.shape[-1] < im_size:
        # pad with zeros
        pad_width = ((0,0), (0,0), (0,0), (im_size // 4, im_size // 4 + 1))
        im_array = np.pad(im_array, pad_width, mode='constant', constant_values=0.0)
        
    if im_array.shape[-2] < im_size:
        # pad with zeros
        pad_width = ((0,0), (0,0), (im_size // 4, im_size // 4 + 1), (0,0))
        im_array = np.pad(im_array, pad_width, mode='constant', constant_values=0.0)
        
    if im_array.shape[-1] > im_size:
        # Crop on axis=-1
        crop_amount = im_array.shape[-1] - im_size
        im_array = im_array[:, :, :, crop_amount // 2 : - crop_amount // 2]
        
    if im_array.shape[-2] > im_size:
        # Crop on axis=-2
        crop_amount = im_array.shape[-2] - im_size
        im_array = im_array[:, :, crop_amount // 2 : - crop_amount // 2, :]

    return im_array

def organize_image_backgrounds(im_dir, image_bank_size, config_dicts):
    """
    Sort image files based on map. If no map exists, sort randomly.

    :param im_dir: path to directory of images
    :param image_bank_size: number of images in user-specified bank
    :param config_dicts: list of config_dicts
    :return: image_indices: the indices of the images utilized for each config_dict
    """
    map_columns = []
    if os.path.exists(im_dir + '/' + 'map.txt'):
        # Read the map
        df = pd.read_csv(im_dir + '/' + 'map.txt', delim_whitespace=True)

        # Trim to just the columns in the config dict
        map_columns = [x for x in df.columns if x in config_dicts[0].keys()]
        
    if len(map_columns) == 0:
        # Sort randomly
        image_indices = np.random.choice(np.arange(image_bank_size), replace=True, size=len(config_dicts))
    
    else:
        # Trim df to just the columns needed
        map_param_array = df[map_columns].values[:, np.newaxis]
        
        # for each entry in config_dict, set up numpy broadcasting
        im_param_array = []
        for config_dict in config_dicts:
            im_param_array.append([config_dict[x] for x in map_columns])
        im_param_array = np.array(im_param_array)

        # divide by stds to put parameters on same footing
        im_stds = np.std(im_param_array, axis=0)[np.newaxis, :]
        im_stds = np.where(im_stds < 1.0, np.ones(len(im_stds)), im_stds)

        # Find the closest image to each parameter combination in map_param_array
        image_indices = np.argmin(np.sum(np.abs(im_param_array - map_param_array) / im_stds, axis=2), axis=0)

    return image_indices
    

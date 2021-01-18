"""Special functions, useful for implementing correlations in sampled parameters.
Utilizing the special keyword is discouraged, but makes it possible to update 
all values drawn from a distribution prior to image generation.
"""

import deeplenstronomy.distributions as distributions

# Examples of ways where a user can implement their own relationships not already
# in deeplenstronomy

def brighten_everything(output_dict, light_profile_mag, bands):
    """
    Brighten everything in a light profile by a given number of mags.

    Args:
        output_dict (dict): flat dictionary being used to simulate images
        light_profile_mag (str): light profile id to be recolored + '-' + mags to brighten. E.g. 'LIGHT_PROFILE_1-2.5'
        bands (str): comma-separated string of bands used
    
    Returns:
        output_dict: the same dictionary with some overwritten values
    """

    light_profile = light_profile_mag.split('-')[0]
    mag = float(light_profile_mag.split('-')[1])
    
    for band, sim_dict in output_dict.items():
        for k in sim_dict.keys():
            if k.find(light_profile + '-magnitude') != -1:
                output_dict[band][k] = output_dict[band][k] - mag

    return output_dict


def make_blueer(output_dict, light_profile_mag, bands):
    """
    Brighten the blue bands in a survey, using des, delve, and lsst as examples.

    Args:
        output_dict (dict): flat dictionary being used to simulate images
        light_profile_mag (str): light profile id to be recolored + '-' + mags to brighten. E.g. 'LIGHT_PROFILE_1-2.5'
        bands (str): comma-separated string of bands used
    
    Returns: 
        output_dict: the same dictionary with some overwritten values
    """
    # brighten g and r

    light_profile = light_profile_mag.split('-')[0]
    mag = float(light_profile_mag.split('-')[1])
    mag_correction = {'g': mag, 'r': 0.5 * mag, 'i': 0.0, 'z': 0.0, 'Y': 0.0}

    for band, sim_dict in output_dict.items():
        for k in sim_dict.keys():
            if k.find(light_profile + '-magnitude') != -1:
                output_dict[band][k] = output_dict[band][k] - mag_correction[band]

    return output_dict

def make_redder(output_dict, light_profile_mag, bands):
    """
    Brighten the red bands in a survey, using des, delve, and lsst as examples.

    Args:
        output_dict (dict): flat dictionary being used to simulate images
        light_profile_mag (str): light profile id to be recolored + '-' + mags to brighten. E.g. 'LIGHT_PROFILE_1-2.5'
        bands (str): comma-separated string of bands used
    
    Returns: 
        output_dict: the same dictionary with some overwritten values
    """
    # brighten i, z, and Y
    
    light_profile = light_profile_mag.split('-')[0]
    mag = float(light_profile_mag.split('-')[1])
    mag_correction = {'g': 0.0, 'r': 0.0, 'i': 0.5 * mag, 'z': mag, 'Y': 1.5 * mag}

    for band, sim_dict in output_dict.items():
        for k in sim_dict.keys():
            if k.find(light_profile + '-magnitude') != -1:
                output_dict[band][k] = output_dict[band][k] - mag_correction[band]

    return output_dict


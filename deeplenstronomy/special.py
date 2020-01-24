# Special functions, useful for implementing correlations in sampled parameters

import deeplenstronomy.distributions as distributions

def fundamental_plane(output_dict, profiles, bands):
    """
    Based on the fundamental plane relations, overwrite sampled parameters
    to force desired correlations

    :param output_dict: flat dictionary being used to simulate images
    :param profiles: light and mass profile ids, ie. LIGHT_PROFILE_1-MASS_PROFILE_1
    :param bands: comma-separated string of bands used
    :return: output_dict: the same dictionary with some overwritten values
    """

    print(light_profile)

    return output_dict
    
    
def des_color(output_dict, light_profile, bands):
    """
    Choose adjacent band colors from DES Y3 Gold data

    :param output_dict: flat dictionary being used to simulate images 
    :param light_profile: light profile id to be recolored
    :param bands: comma-separated string of bands used  
    :return: output_dict: the same dictionary with some overwritten values 
    """

    g_mag = distributions.des_mag('g')[0]
    r_mag = g_mag - distributions._des_mag_color('g-r')
    i_mag = r_mag - distributions._des_mag_color('r-i')
    z_mag = i_mag - distributions._des_mag_color('i-z')
    Y_mag = z_mag # kludge

    mags = {'g': g_mag, 'r': r_mag, 'i': i_mag, 'z': z_mag, 'Y': Y_mag}

    # This will only run for the bands in the output dict
    for band, sim_dict in output_dict.items():
        for k in sim_dict.keys():
            if k.find(light_profile + '-magnitude') != -1:
                output_dict[band][k] = mags[band]
        
    return output_dict
    

def brighten_everything(output_dict, light_profile_mag, bands):
    #brighted everything by 2 mag

    light_profile = light_profile_mag.split('-')[0]
    mag = float(light_profile_mag.split('-')[1])
    
    for band, sim_dict in output_dict.items():
        for k in sim_dict.keys():
            if k.find(light_profile + '-magnitude') != -1:
                output_dict[band][k] = output_dict[band][k] - mag

    return output_dict


def make_blueer(output_dict, light_profile_mag, bands):
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
    # brighten i, z, and Y
    
    light_profile = light_profile_mag.split('-')[0]
    mag = float(light_profile_mag.split('-')[1])
    mag_correction = {'g': 0.0, 'r': 0.0, 'i': 0.5 * mag, 'z': mag, 'Y': 1.5 * mag}

    for band, sim_dict in output_dict.items():
        for k in sim_dict.keys():
            if k.find(light_profile + '-magnitude') != -1:
                output_dict[band][k] = output_dict[band][k] - mag_correction[band]

    return output_dict

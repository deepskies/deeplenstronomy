# A module to add galaxy paramters to merged lcs

import glob
import numpy as np
import pandas as pd
import sys
import numpy as np
import utils

def append_gal_params(lc_file, param_dict, suffix):
    lc = pd.read_csv(lc_file)

    for param, value in param_dict.items():
        lc[param] = value

    z_lens = pick_z_lens(lc['z_source'].values[0])
    lc['z_lens'] = z_lens
    #lc['theta_E'] = get_theta_E(lc['z_source'].values[0], z_lens)

    lc.to_csv('lcs_plus_gal_param/' + lc_file[11:-3] + '_' + suffix + '.csv')
    return


def pick_z_lens(z_source):
    z_lens = np.random.uniform(low=0.05, high=z_source - 0.01)
    return z_lens

"""
def get_theta_E(z_source, z_lens, lens_mass=2.3e22):
    assert z_source > z_lens
    
    DL = utils.convert_z_to_d(np.array([z_source, z_lens]))
    DLs = DL[0]
    DLl = DL[1]   # <-- luminosity distances
    
    DAs = DLs / (1 + z_source)**2
    DAl = DLl / (1 + z_lens)**2   # <-- angular diameter distances
    
    distance_ratio = (DAs - DAl) / (DAs * DAl) #units of pc-1
    distance_ratio_meters = distance_ratio / 3.086e16 #units of m-1
    
    G = 6.67408e-11 #units m3 kg-1 s-2
    c = 299792458.0 #units m s-1
    Msolar = 1.989e30 #units kg
    M = lens_mass * Msolar
    
    theta_E_squared = 4 * G * M * distance_ratio_meters / c**2
    
    return np.sqrt(theta_E_squared)
""" 

param_dicts = np.load('galaxy_params.npy').item()

#param_dicts = {'001':{'param1': param1_value,
#                     'param2': param2_value,
#                     ...
#                     }
#              '002':{'param1': param1_value,
#                     'param2': param2_value,
#                     ...
#                     }
#              ...
#              }



lc_files = glob.glob('merged_lcs/*.lc')
counter = 0.0
total = len(lc_files)


for lc_file in lc_files:
    #track progress
    counter += 1.0
    progress = counter / total * 100
    sys.stdout.write('\rProgress:  %.2f %%' %progress)
    sys.stdout.flush()

    #choose random galaxy geometry
    param_key = np.random.choice(list(param_dicts.keys()), size=1, p=np.ones(len(list(param_dicts.keys()))) / len(list(param_dicts.keys())))[0]
    param_dict = param_dicts[param_key]

    append_gal_params(lc_file, param_dict, param_key)

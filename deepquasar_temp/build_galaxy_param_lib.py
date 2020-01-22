import numpy as np

def make_single_param_dict():

    e1_min, e1_max = -0.4, 0.4
    e2_min, e2_max = -0.4, 0.4
    theta_E_min, theta_E_max = 0.5, 3.0
    R_sersic_min, R_sersic_max = 0.5, 5.0
    n_sersic_min, n_sersic_max = 0.5, 10.0
    lens_mag_min, lens_mag_max = 18, 21
    source_mag_min, source_mag_max = 19, 23
    source_center_x_min, source_center_x_max = -1.0, 1.0
    source_center_y_min, source_center_y_max = -1.0, 1.0
    shear_e1_min, shear_e1_max = -0.05, 0.05
    shear_e2_min, shear_e2_max = -0.05, 0.05
    
    #note: z_lens, z_source, and theta_E are assigned separately

    param_dict = {'lens_e1': np.random.uniform(low=e1_min, high=e1_max),
                  'lens_e2': np.random.uniform(low=e2_min, high=e2_max),
                  'lens_R_sersic': np.random.uniform(low=R_sersic_min, high=R_sersic_max),
                  'lens_n_sersic': np.random.uniform(low=n_sersic_min, high=n_sersic_max),
                  'lens_mag': np.random.uniform(low=lens_mag_min, high=lens_mag_max),
                  'lens_shear_e1': np.random.uniform(low=shear_e1_min, high=shear_e1_max),
                  'lens_shear_e2': np.random.uniform(low=shear_e2_min, high=shear_e2_max),
                  'lens_theta_E': np.random.uniform(low=theta_E_min, high=theta_E_max),
                  
                  'source_e1': np.random.uniform(low=e1_min, high=e1_max),
                  'source_e2': np.random.uniform(low=e2_min, high=e2_max),
                  'source_R_sersic': np.random.uniform(low=R_sersic_min, high=R_sersic_max),
                  'source_n_sersic': np.random.uniform(low=n_sersic_min, high=n_sersic_max),
                  'source_mag': np.random.uniform(low=source_mag_min, high=source_mag_max),
                  'source_center_x': np.random.uniform(low=source_center_x_min, high=source_center_x_max),
                  'source_center_y': np.random.uniform(low=source_center_y_min, high=source_center_y_max)}



    return param_dict

def make_library():
    
    keys = np.arange(10000, dtype=int)
    str_keys = [str(x) for x in keys]

    keys = []
    for key in str_keys:
        if len(key) == 1:
            keys.append('000' + key)
        elif len(key) == 2:
            keys.append('00' + key)
        elif len(key) == 3:
            keys.append('0' + key)
        elif len(key) == 4:
            keys.append(key)
        else:
            print("Unexpected key value: ", key)

    param_dicts = {}
    for key in keys:
        param_dicts[key] = make_single_param_dict()
        
    return param_dicts

param_dicts = make_library()


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




np.save('galaxy_params.npy', param_dicts)

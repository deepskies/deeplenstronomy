# Helper functions

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

# Helper functions and classes

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

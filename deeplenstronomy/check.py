# A module to check for user errors in the main config file

from deeplenstronomy.utils import KeyPathDict 
import sys

class AllChecks():
    """
    Define new checks as methods starting with 'check_'
    Methods must return err_code, err_message where
    err_code == 0 means success and err_code != 0 means failure
    If failure, the err_message is printed and sys.exit() is called
    """
    
    def __init__(self, full_dict, config_dict):
        """
        Trigger the running of all checks
        """
        # convert to benedict objects for easier parsing
        kp_f = KeyPathDict(full_dict, keypath_separator='.')
        self.full = kp_f
        self.full_keypaths = kp_f.keypaths()
        kp_c = KeyPathDict(config_dict, keypath_separator='.')
        self.config = kp_c
        self.config_keypaths = kp_c.keypaths()

        # find all check functions
        self.checks = [x for x in dir(self) if x.find('check_') != -1]

        # run checks
        for check in self.checks:
            err_code, err_message = eval('self.' + check + '()')
            if err_code != 0:
                print(err_message)
                sys.exit()

        return

    def check_top_level_existence(self):
        for name in ['DATASET', 'SURVEY', 'IMAGE', 'COSMOLOGY', 'SPECIES', 'GEOMETRY']:
            if name not in self.full.keys():
                return 1, "Missing {0} section from config file".format(name)
        return 0, "passed"


def run_checks(full_dict, config_dict):
    """
    Instantiate an AllChecks object to run checks

    :param full_dict: a Parser.full_dict object
    :param config_dict: a Parser.config_dict object
    """
    check_runner = AllChecks(full_dict, config_dict)


        

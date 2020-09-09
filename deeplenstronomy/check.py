# A module to check for user errors in the main config file

from deeplenstronomy.utils import KeyPathDict 
import sys

class ConfigFileError(Exception): pass

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
        # convert to KeyPathDict objects for easier parsing
        kp_f = KeyPathDict(full_dict, keypath_separator='.')
        self.full = kp_f
        self.full_keypaths = kp_f.keypaths()
        kp_c = KeyPathDict(config_dict, keypath_separator='.')
        self.config = kp_c
        self.config_keypaths = kp_c.keypaths()

        # find all check functions
        self.checks = [x for x in dir(self) if x.find('check_') != -1]

        # run checks
        total_errs = []
        for check in self.checks:

            try:
                err_messages = eval('self.' + check + '()')
            except Exception:
                err_messages = ["CheckFunctionError: " + check] 

            total_errs += err_messages

        # report errors to user
        if len(total_errs) != 0:
            kind_output(total_errs)
            raise ConfigFileError

        return

    ### Helper methods
    @staticmethod
    def config_dict_format(*args):
        return "['" + "']['".join(list(args)) + "']"

    def config_lookup(self, lookup_str):
        return eval("self.config_dict" + lookup_str)
    
    ### Check functions
    def check_top_level_existence(self):
        errs = []
        for name in ['DATASET', 'SURVEY', 'IMAGE', 'COSMOLOGY', 'SPECIES', 'GEOMETRY']:
            if name not in self.full.keys():
                errs.append("Missing {0} section from config file".format(name))
        return errs

    def check_low_level_existence(self):
        errs = []
        param_names = {"DATASET.NAME",
                       "DATASET.PARAMETERS.SIZE",
                       "COSMOLOGY.PARAMETERS.H0",
                       "COSMOLOGY.PARAMETERS.Om0",
                       "IMAGE.PARAMETERS.exposure_time",
                       "IMAGE.PARAMETERS.numPix",
                       "IMAGE.PARAMETERS.pixel_scale",
                       "IMAGE.PARAMETERS.psf_type",
                       "IMAGE.PARAMETERS.read_noise",
                       "IMAGE.PARAMETERS.ccd_gain",
                       "SURVEY.PARAMETERS.BANDS",
                       "SURVEY.PARAMETERS.seeing",
                       "SURVEY.PARAMETERS.magnitude_zero_point",
                       "SURVEY.PARAMETERS.sky_brightness",
                       "SURVEY.PARAMETERS.num_exposures"}
        for param in param_names:
            try:
                config_obj = self.config_lookup(self.config_dict_format(param.split('.')))
            except KeyError:
                errs.append(param + "is missing from the Config File")

        return errs

    def check_not_allowed_to_be_drawn_from_a_distribution(self):
        errs = []
        param_names = {"DATASET.NAME",
                       "DATASET.PARAMETERS.SIZE",
                       "DATASET.PARAMETERS.OUTDIR",
                       "IMAGE.PARAMETERS.numPix",
                       "COSMOLOGY.PARAMETERS.H0",
                       "COSMOLOGY.PARAMETERS.Om0",
                       "COSMOLOGY.PARAMETERS.Tcmb0",
                       "COSMOLOGY.PARAMETERS.Neff",
                       "COSMOLOGY.PARAMETERS.m_nu",
                       "COSMOLOGY.PARAMETERS.Ob0"}
        for param in param_names:
            try:
                config_obj = self.config_lookup(self.config_dict_format(param.split('.')))
            except KeyError:
                # The checked parameter was not in the config dict
                continue
            
            if isinstance(config_obj, dict):
                errs.append(param + " cannot be drawn from a distribution")
        return errs

    def check_valid_geometry(self):
        errs = []

        # Check keys
        detected_configurations = []
        fractions = []
        for k in self.config_dict['GEOMETRY'].keys():
            if not k.startswith('CONFIGURATION_'):
                errs.append('GEOMETRY.' + k + ' is an invalid Config File entry')

            try:
                val = int(k.split('_')[-1])
                if val < 1:
                    errs.append('GEOMETRY.' + k + ' is an invalid Config File entry')
                detected_configurations.append(val)
            except TypeError:
                errs.append('GEOMETRY.' + k + ' needs a valid integer index greater than zero')

            if "FRACTION" not in self.config_dict['GEOMETRY'][k].keys():
                errs.append("GEOMETRY." + k " .FRACTION is missing")
            else:
                try:
                    fraction = float(self.config_dict['GEOMETRY'][k]['FRACTION'])
                    fractions.append(fraction)
                except TypeError:
                    errs.append("GEOMETRY." + k " .FRACTION must be a float")

        if len(detected_configurations) != max(detected_configurations):
            errs.append("CONFIGURATIONs in the GEOMETRY section must be indexed as 1, 2, 3, ...")

        if not (0.0 < sum(fractions) <= 1.0):
            errs.append("CONFIGURATION FRACTIONs must sum to a number between 0.0 and 1.0")
                
        return errs
    # End check functions

def kind_output(errs):
    """
    Print all detected errors in the configuration file to the screen
    """
    for err in errs:
        print(err)
    return


def run_checks(full_dict, config_dict):
    """
    Instantiate an AllChecks object to run checks

    :param full_dict: a Parser.full_dict object
    :param config_dict: a Parser.config_dict object
    """
    try:
        check_runner = AllChecks(full_dict, config_dict)
    except ConfigFileError:
        print("Fatal error(s) detected in config file. Please edit and rerun.")
        sys.exit()

        

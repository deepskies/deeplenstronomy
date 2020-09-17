# A module to check for user errors in the main config file

import glob
from inspect import getfullargspec
import os
import sys

from astropy.io import fits
import pandas as pd

from deeplenstronomy.utils import KeyPathDict
import deeplenstronomy.distributions as distributions

class ConfigFileError(Exception): pass

class AllChecks():
    """
    Define new checks as methods starting with 'check_'
    Methods must return a list of err_message where
    an empty list means success and a nonempty list means failure
    If failure, the err_messages are printed and sys.exit() is called
    """
    
    def __init__(self, full_dict, config_dict):
        """
        Trigger the running of all checks
        """
        # flag for already checked timeseries files
        self.checked_ts_bands = False
        
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

    def config_lookup(self, lookup_str, full=False):
        if not full:
            return eval("self.config_dict" + lookup_str)
        else:
            return eval("self.full_dict" + lookup_str)
        
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

    def check_for_auxiliary_files(self):
        errs = []
        input_paths = [x for x in self.full_keypaths if x.find("INPUT") != -1]
        input_files = [self.config_lookup(self.config_dict_format(param.split('.')), full=True) for param in input_paths]
        for filename in input_files:
            if not os.path.exists(filename):
                errs.append("Unable to find auxiliary file: " + filename)
        return errs

    def check_for_valid_distribution_entry(self):
        errs = []
        distribution_paths = [x for x in self.full_keypaths if x.find("DISTRIBUTION.") != -1]
        distribution_dicts = [self.config_lookup(self.config_dict_format(param.split('.'))) for param in distribution_paths]
        for distribution_dict, path in zip(distribution_dicts, distribution_paths):
            # must have name key - return early to not break the remaining parts of this function
            if "NAME" not in distribution_dict.keys():
                errs.append(path + " is missing the NAME key")
                return errs
            else:
                # name must be valid
                if distribution_dict["NAME"] not in dir(distributions):
                    errs.append(path + " is not a valid distribution name")
                    return errs

            # must have parameter key
            if "PARAMETERS" not in distribution_dict.keys():
                errs.append(path + " is missing the PARAMETERS key")
            else:
                # parameters must be valid for the distribution
                allowed_params = getfullargspec(eval("distributions." + distribution_dict["NAME"]))[0]
                for param in distribution_dict["PARAMETERS"]:
                    if param not in allowed_params:
                        errs.append(path + '.PARAMETERS.' + param + ' is not in the allowed list of ({0}) for the distribtuion '.format(', '.join(allowed_params)) + distribution_dict["NAME"]) 
        return errs
    
    def check_input_distributions(self):
        errs = []
        if "DISTRIBUTIONS" in self.config_dict.keys():
            # there must be at least 1 USERDIST_ key
            userdists = [x for x in self.config_dict["DISTRIBUTIONS"].keys() if x.startswith("USERDIST_")]
            if len(userdists) == 0:
                errs.append("DISTRIBUTIONS section must have at least 1 USERDIST key")
            else:
                for userdist in userdists:
                    # name must be a single value
                    if not isinstance(self.config_dict["DISTRIBUTIONS"][userdist], str):
                        errs.append("DISTRIBUTIONS." + userdist + " must be a single name of a file")
                    else:
                        # specified file must exist
                        if not os.path.exists(self.config_dict["DISTRIBUTIONS"][userdist]):
                            errs.append("DISTRIBUTIONS." + userdist + "File '" + self.config_dict["DISTRIBUTIONS"][userdist] + "' not found")
                        else:
                            # must be able to read file
                            df = None
                            try:
                                df = pd.read_csv(filename, delim_whitespace=True)
                                if "WEIGHT" not in df.columns:
                                    errs.append("WEIGHT column not found in  DISTRIBUTIONS." + userdist + "File '" + self.config_dict["DISTRIBUTIONS"][userdist] + "'")
                            except Exception:
                                errs.append("Error reading DISTRIBUTIONS." + userdist + "File '" + self.config_dict["DISTRIBUTIONS"][userdist] + "'")
                            finally:
                                del df
        return errs

    def check_image_backgrounds(self):
        errs = []
        if "BACKGROUNDS" in self.config_dict.keys():
            # value must be a single value
            if not isinstance(self.config_dict["BACKGROUNDS"], str):
                errs.append("BACKGROUNDS must be a single value")
            else:
                # directory must exist
                if not os.path.exists(self.config_dict["BACKGROUNDS"]):
                    errs.append("BACKGROUNDS directory '" + self.config_dict["BACKGROUNDS"] + "' not found")
                else:
                    dimensions = {}
                    # one file must exist per band
                    for band in self.config_dict["SURVEY"]["PARAMETERS"]["BANDS"].split(','):
                        if not os.path.exists(self.config_dict["BACKGROUNDS"] + "/" + band + ".fits"):
                            errs.append("BACKGROUNDS: " + self.config_dict["BACKGROUNDS"] + '/' + band + ".fits is missing")
                        else:
                            # must be able to open file
                            hdu, data = None, None
                            try:
                                hdu = fits.open(self.config_dict["BACKGROUNDS"] + '/' + band + '.fits')
                                data = hdu[0].data
                                if len(data.shape) != 3:
                                    errs.append("image data in " + self.config_dict["BACKGROUNDS"] + '/' + band + '.fits is formatted incorrectly')
                                dimensions[band] = data.shape[0]
                            except Exception:
                                errs.append("Error reading " + self.config_dict["BACKGROUNDS"] + '/' + band + '.fits')
                            finally:
                                if hdu is not None:
                                    hdu.close()
                                del data

                    # map.txt file is formatted correctly
                    if os.path.exists(self.config_dict["BACKGROUNDS"] + '/map.txt'):
                        df = None
                        try:
                            df = pd.read_csv(self.config_dict["BACKGROUNDS"] + '/' + 'map.txt', delim_whitespace=True)
                            dimensions["map"] = df.shape[0]
                        except Exception:
                            err.append("Error reading " + self.config_dict["BACKGROUNDS"] + '/map.txt')
                        finally:
                            del df

                    # dimensions of images and (optional) map must be the same
                    if len(set(dimensions.values())) != 1:
                        errs.append("BACKGROUNDS: dimensions of images files and possibly map.txt are inconsistent")

        return errs
    
    def _valid_model(self, model_name, path):
        errs = []

        # check that transmission curves exist for the bands
        if model_name not in ['flat', 'flatnoise', 'variable', 'variablenoise']:
            if not self.checked_ts_bands:
                for band in self.config_dict["SURVEY"]["PARAMETERS"]["BANDS"].split(','):
                    try:
                        filter_file = [x for x in glob.glob('filters/*_' + band + '.*')][0]
                        passband = pd.read_csv(filter_file,
                                               names=['WAVELENGTH', 'TRANSMISSION'],
                                               delim_whitespace=True, comment='#')
                    except Exception:
                        errs.append("Unable to find transmission curve for " + band + " in the filters/ directory")
                self.checked_ts_bands = True

            # check that the model name is allowed
            try:
                obj = model_name.split('_')[0]
                sed = model_name.split('_')[1]
            except IndexError:
                errs.append(path + '.' + model_name + ' is an invalid timeseries model')
                obj, sed = 'ia', 'random'

            if obj == 'ia':
                if sed not in ['random', 'salt2_template_0.dat', 'salt2_template_0.dat']:
                    errs.append(path + '.' + model_name + ' does not have a valid sed specified')
            elif obj == 'cc':
                if sed not in ['random', 'Nugent+Scolnic_IIL.SED', 'SNLS-04D1la.SED', 'SNLS-04D4jv.SED',
                               'CSP-2004fe.SED', 'CSP-2004gq.SED', 'CSP-2004gv.SED', 'CSP-2006ep.SED',
                               'CSP-2007Y.SED', 'SDSS-000018.SED', 'SDSS-000020.SED', 'SDSS-002744.SED',
                               'SDSS-003818.SED', 'SDSS-004012.SED', 'SDSS-012842.SED', 'SDSS-013195.SED',
                               'SDSS-013376.SED', 'SDSS-013449.SED', 'SDSS-014450.SED', 'SDSS-014475.SED',
                               'SDSS-014492.SED', 'SDSS-014599.SED', 'SDSS-015031.SED', 'SDSS-015320.SED',
                               'SDSS-015339.SED', 'SDSS-015475.SED', 'SDSS-017548.SED', 'SDSS-017564.SED',
                               'SDSS-017862.SED', 'SDSS-018109.SED', 'SDSS-018297.SED', 'SDSS-018408.SED',
                               'SDSS-018441.SED', 'SDSS-018457.SED', 'SDSS-018590.SED', 'SDSS-018596.SED',
                               'SDSS-018700.SED', 'SDSS-018713.SED', 'SDSS-018734.SED', 'SDSS-018793.SED',
                               'SDSS-018834.SED', 'SDSS-018892.SED', 'SDSS-019323.SED', 'SDSS-020038.SED']:
                    errs.append(path + '.' + model_name + ' does not have a valid sed specified')
            elif obj == 'user':
                if not os.path.exists("seds/user/" + sed):
                    errs.append(path + '.' + model_name + ' sed file ({0}) is missing'.format(sed))
                else:
                    # check that the file can be opened properly
                    try:
                        df = pd.read_csv(sed_filename,
                                         names=['NITE', 'WAVELENGTH_REST', 'FLUX'],
                                         delim_whitespace=True, comment='#')
                    except Exception:
                        errs.append(path + "." + model_name + " sed file ({0})) could not be read properly.".format(sed))
            else:
                errs.append(path + '.' + model_name + ' is an invalid timeseries model')
                
        return errs

    def _valid_galaxy(self, k):
        errs, names = [], []

        # Must have a name key
        if "NAME" not in self.config_dict['SPECIES'][k].keys():
            errs.append("SPECIES." + k + " is missing an entry for NAME")
        else:
            # name must be a string
            if not isinstance(self.config_dict['SPECIES'][k]["NAME"], str):
                errs.append("SPECIES." + k + ".NAME must be the name of a function in distribution.py")
            else:
                names.append(self.config_dict['SPECIES'][k]["NAME"])

        # Check LIGHT_PROFILEs, MASS_PROFILEs, and SHEAR_PROFILEs
        detected_light_profiles, detected_mass_profiles, detected_shear_profiles = [], [], []
        for profile_k in self.config_dict['SPECIES'][k].keys():
            if profile_k.startswith('LIGHT_PROFILE_') or profile_k.startswith('MASS_PROFILE_') or profile_k.startswith('SHEAR_PROFILE_'):
                # Index must be valid
                detections, errors = self._valid_index(profile_k, "SPECIES." + k)
                if profile_k.startswith('LIGHT_PROFILE_'):
                    detected_light_profiles += detections
                elif profile_k.startswith('MASS_PROFILE_'):
                    detected_mass_profiles += detections
                elif profile_k.startswith('SHEAR_PROFILE_'):
                    detected_shear_profiles += detections
                errs += errors

                # Must have name and parameters
                if "NAME" not in self.config_dict['SPECIES'][k][profile_k].keys():
                    errs.append("SPECIES." + k + "." + profile_k + " needs a NAME")
                else:
                    if not isinstance(self.config_dict['SPECIES'][k][profile_k]["NAME"], str):
                        errs.append("SPECIES." + k + "." + profile_k + ".NAME must be a single name")
                if "PARAMETERS" not in self.config_dict['SPECIES'][k][profile_k].keys():
                    errs.append("SPECIES." + k + "." + profile_k + " needs PARAMETERS")
                else:
                    if not isinstance(self.config_dict['SPECIES'][k][profile_k]["PARAMETERS"], dict):
                        errs.append("SPECIES." + k + "." + profile_k + ".PARAMETERS must contain all parameters for the lenstronomy profile")

            # If MODEL is specified, it must be valid
            if profile_k == "MODEL":
                if not isinstance(self.config_dict['SPECIES'][k][profile_k], str):
                    errs.append("SPECIES." + k + "." + profile_k + ".MODEL must be a single name")
                else:
                    errs += self._valid_model(self.config_dict['SPECIES'][k][profile_k], "SPECIES." + k + "." + profile_k)

        # need at least one light profile
        if len(detected_light_profiles) < 1:
            errs.append("SPECIES." + k + " needs at least one LIGHT_PROFILE")
        # all indexing must be valid
        elif len(detected_light_profiles) != max(detected_light_profiles):
            errs.append("SPECIES." + k + " LIGHT_PROFILEs must be indexed as 1, 2, 3 ...")
        if len(detected_mass_profiles) > 0 and len(detected_mass_profiles) != max(detected_mass_profiles):
            errs.append("SPECIES." + k + " MASS_PROFILEs must be indexed as 1, 2, 3 ...")
        if len(detected_shear_profiles) > 0 and len(detected_shear_profiles) != max(detected_shear_profiles):
            errs.append("SPECIES." + k + " SHEAR_PROFILEs must be indexed as 1, 2, 3 ...")
            
        return errs, names

    def _valid_point_source(self, k):
        errs, names = [], []
        # Must have name key
        if "NAME" not in self.config_dict['SPECIES'][k].keys():
            errs.append("SPECIES." + k + " is missing an entry for NAME")
        else:
            # name must be a string
            if not isinstance(self.config_dict['SPECIES'][k]["NAME"], str):
                errs.append("SPECIES." + k + ".NAME must be a sinlge unique value")
            else:
                names.append(self.config_dict['SPECIES'][k]["NAME"])

        # Must have a host key
        if "HOST" not in self.config_dict['SPECIES'][k].keys():
            errs.append("SPECIES." + k + " must have a valid HOST")
        else:
            # host name must be a single value
            if not isinstance(self.config_dict['SPECIES'][k]["HOST"], str):
                errs.append("SPECIES." + k + ".HOST must be a single name")
            elif self.config_dict['SPECIES'][k]["HOST"] == "Foreground":
                pass
            else:
                # host must appear in SPECIES section
                if len([x for x in self.config if x.startswith("SPECIES.") and x.find("NAME." + self.config_dict['SPECIES'][k]["HOST"]) != -1]) == 0:
                    errs.append("HOST for SPECIES." + k + " is not found in SPECIES section")

        # Must have PARAMETERS
        if "PARAMETERS" not in self.config_dict['SPECIES'][k].keys():
            errs.append("SPECIES." + k + " must have PARAMETERS")
        else:
            if not isinstance(self.config_dict['SPECIES'][k]["PARAMETERS"], dict):
                errs.append("SPECIES." + k + ".PARAMETERS must be a dictionary")
            else:
                # separation must be used properly
                if "sep" in self.config_dict['SPECIES'][k]["PARAMETERS"].keys():
                    # sep unit must be specified
                    if "sep_unit" not in self.config_dict['SPECIES'][k]["PARAMETERS"].keys():
                        errs.append("sep is specified for SPECIES." + k + ".PARAMETERS but sep_unit is missing")
                    else:
                        if not isinstance(self.config_dict['SPECIES'][k]["PARAMETERS"]["sep_unit"], str):
                            errs.append("SPECIES." + k + ".PARAMETERS.sep_unit must be either 'arcsec' or 'kpc'")
                        else:
                            if self.config_dict['SPECIES'][k]["PARAMETERS"]["sep_unit"] not in ['arcsec', 'kpc']:
                                errs.append("SPECIES." + k + ".PARAMETERS.sep_unit must be either 'arcsec' or 'kpc'")

                # magnitude must be one of the parameters
                if "magnitude" not in self.config_dict['SPECIES'][k]["PARAMETERS"].keys():
                    errs.append("SPECIES." + k + ".PARAMETERS.magnitude must be specified")

        # If timeseries model is specified, it must be a valid model
        if "MODEL" in self.config_dict['SPECIES'][k].keys():
            if not isinstance(self.config_dict['SPECIES'][k]["MODEL"], str):
                errs.append("SPECIES." + k + ".MODEL must be a single name")
            else:
                errs += self._valid_model(self.config_dict['SPECIES'][k]["MODEL"], "SPECIES." + k + '.MODEL')
                    
        return errs, names

    def _valid_noise(self, k):
        errs, names = [], []
        # Must have name key
        if "NAME" not in self.config_dict['SPECIES'][k].keys():
            errs.append("SPECIES." + k + " is missing an entry for NAME")
        else:
            # name must be a string 
            if not isinstance(self.config_dict['SPECIES'][k]["NAME"], str):
                errs.append("SPECIES." + k + ".NAME must be the name of a function in distribution.py")
            else:
                names.append(self.config_dict['SPECIES'][k]["NAME"])

            # name must be a valid distribution
            if self.config_dict['SPECIES'][k]["NAME"].lower() not in dir(distributions):
                errs.append("SPECIES." + k + ".NAME must be the name of a function in distribution.py")

        # Must have parameter key
        if "PARAMETERS" not in self.config_dict['SPECIES'][k].keys():
            errs.append("SPECIES." + k + " is missing an entry for PARAMETERS")

        return errs, names

    def _valid_index(self, k, path):
        detections, errs = [], []
        try:
            val = int(k.split('_')[-1])
            detections.append(val)
        except TypeError:
            errs.append(path + '.' + k + ' must be indexed with a valid integer')
        return detections, errs
    
    def check_valid_species(self):
        errs, names = [], []

        # There must be at least one species
        if len(list(self.config_dict['SPECIES'].keys())) == 0:
            errs.append("SPECIES sections needs at least one SPECIES")

        # Check keys
        detected_galaxies, detected_point_sources, detected_noise_sources = [], [], []
        for k in self.config_dict['SPECIES'].keys():
            detections, errors = self._valid_index(k, "SPECIES")
            errs += errors
            
            if k.startswith('GALAXY_'):
                detected_galaxies += detections
                errors, obj_names = self._valid_galaxy(k)
                errs += errors
                names += obj_names
            elif k.startswith('POINTSOURCE_'):
                detected_point_sources += detections
                errors, obj_names = self._valid_point_source(k)
                errs +=	errors
                names += obj_names
            elif k.startswith('NOISE_'):
                detected_noise_sources += detections
                errors, obj_names = self._valid_noise(k)
                errs +=	errors
                names += obj_names
            else:
                # unexpected entry
                errs.append(k + " in SPECIES is an invalid entry")

        # each class must be indexed sequentially
        if len(detected_galaxies) != max(detected_galaxies):
            errs.append('GALAXY objects in SPECIES must be indexed like 1, 2, 3, ...')
        if len(detected_point_sources) != max(detected_point_sources):
            errs.append('POINTSOURCE objects in SPECIES must be indexed like 1, 2, 3, ...')
        if len(detected_noise_sources) != max(detected_noise_sources):
            errs.append('NOISE objects in SPECIES must be indexed like 1, 2, 3, ...')

        # All objects must have a unique name
        if len(set(names)) != len(names):
            errs.append("All entries in SPECIES must have a unique NAME")

        return errs
    
    def check_valid_geometry(self):
        errs = []

        # There must be at least one configuration
        if len(list(self.config_dict['GEOMETRY'].keys())) == 0:
            errs.append("GEOMETRY sections needs at least one CONFIGURATION")
        
        # Check keys
        detected_configurations, detected_noise_sources, fractions = [], [], []
        for k in self.config_dict['GEOMETRY'].keys():
            if not k.startswith('CONFIGURATION_'):
                errs.append('GEOMETRY.' + k + ' is an invalid Config File entry')

            # Configurations must be indexed with a valid integer
            try:
                val = int(k.split('_')[-1])
                if val < 1:
                    errs.append('GEOMETRY.' + k + ' is an invalid Config File entry')
                detected_configurations.append(val)
            except TypeError:
                errs.append('GEOMETRY.' + k + ' needs a valid integer index greater than zero')

            # Every configuration needs a FRACTION that is a valid float
            if "FRACTION" not in self.config_dict['GEOMETRY'][k].keys():
                errs.append("GEOMETRY." + k " .FRACTION is missing")
            else:
                try:
                    fraction = float(self.config_dict['GEOMETRY'][k]['FRACTION'])
                    fractions.append(fraction)
                except TypeError:
                    errs.append("GEOMETRY." + k " .FRACTION must be a float")

            # Configurations must have at least one plane
            if len(list(self.config_dict['GEOMETRY'][k].keys())) == 0:
                errs.append("CEOMETRY." + k + " must have at least one PLANE")

            detected_planes = []
            for config_k in self.config_dict['GEOMETRY'][k].keys():
                # check individual plane properties
                if config_k.startswith('PLANE_'):
                    # Plane index must be a valid integer
                    try:
                        val = int(config_k.split('_')[-1])
                        if val < 1:
                            errs.append('GEOMETRY.' + k + '.' + config_k + ' is an invalid Config File entry')
                        detected_planes.append(val)
                    except TypeError:
                        errs.append('GEOMETRY.' + k + '.' + config_k + ' needs a valid integer index greater than zero')

                    # Plane must have a redshift
                    if 'REDSHIFT' not in config_k in self.config_dict['GEOMETRY'][k][config_k]['PARAMETERS'].keys():
                        errs.append('REDSHIFT is missing from GEOMETRY.' + k + '.' + config_k)

                    detected_objects = []
                    for obj_k in self.config_dict['GEOMETRY'][k][config_k].keys():
                        # check individual object properties
                        if obj_k.startswith('OBJECT_'):
                            # Object index must be a valid integer
                            try:
                                val = int(obj_k.split('_')[-1])
                                if val < 1:
                                    errs.append('GEOMETRY.' + k + '.' + config_k + '.' + obj_k + ' is an invalid Config File entry')
                                detected_objects.append(val)
                            except TypeError:
                                errs.append('GEOMETRY.' + k + '.' + config_k + '.' + obj_k + ' needs a valid integer index greater than zero')

                            # Objects must have a value that appears in the species section
                            if not isinstance(self.config_dict['GEOMETRY'][k][config_k][obj_k], str):
                                errs.append('GEOMETRY.' + k + '.' + config_k + '.' + obj_k + ' must be a single name')

                            species_paths = [x for x in self.config if x.startswith('SPECIES') and x.find('.' + self.config_dict['GEOMETRY'][k][config_k][obj_k] + '.') != -1]
                            if len(species_paths) == 0:
                                errs.append('GEOMETRY.' + k + '.' + config_k + '.' + obj_k + ' is missing from the SPECIES section')
                                
                    # Objects must be indexed sequentially
                    if len(detected_objects) != max(detected_objects):
                        errs.append("OBJECTs in the GEOMETRY." + k + '.' + config_k + " section must be indexed as 1, 2, 3, ...")

                # check noise properties
                elif config_k.startswith('NOISE_SOURCE_'):
                    # index must be a valid integer
                    try:
                        val = int(obj_k.split('_')[-1])
                        if val < 1:
                            errs.append('GEOMETRY.' + k + '.' + config_k + ' is an invalid Config File entry')
                        detected_noise_sources.append(val)
                    except TypeError:
                        errs.append('GEOMETRY.' + k + '.' + config_k + ' needs a valid integer index greater than zero')

                    # Noise sources must have a single value that appears i the species section
                    if not isinstance(self.config_dict['GEOMETRY'][k][config_k], str):
                        errs.append('GEOMETRY.' + k + '.' + config_k + ' must be a single name')

                    species_paths = [x for x in self.config if x.startswith('SPECIES') and x.find('.' + self.config_dict['GEOMETRY'][k][config_k] + '.') != -1]
                    if len(species_paths) == 0:
                        errs.append('GEOMETRY.' + k + '.' + config_k + ' is missing from the SPECIES section')
                        
                # check timeseries properties
                elif config_k == 'TIMESERIES':
                    # Must have objects as keys
                    if "OBJECTS" not in self.config_dict['GEOMETRY'][k][config_k].keys():
                        errs.append("GEOMETRY." + k + ".TIMESERIES is missing the OBJECTS parameter")
                    else:
                        if not isinstance(self.config_dict['GEOMETRY'][k][config_k]["OBJECTS"], list):
                            errs.append("GEOMETRY." + k + ".TIMESERIES.OBJECTS must be a list")
                        else:
                            # listed objects must appear in species section, in the configuration, and have a model defined
                            for obj in self.config_dict['GEOMETRY'][k][config_k]['OBJECTS']:
                                species_paths = [x for x in self.config if x.startswith('SPECIES') and x.find('.' + obj + '.') != -1]
                                if len(species_paths) == 0:
                                    errs.append(obj + "in GEOMETRY." + k + ".TIMESERIES.OBJECTS is missing from the SPECIES section")
                                elif "MODEL" not in config_lookup(config_dict_format(species_paths[0].split('.'))).keys():
                                    errs.append("MODEL for " + obj + " in GEOMETRY." + k + ".TIMESERIES.OBJECTS is missing from the SPECIES section")
                                configuration_paths = [x for x in self.config if x.startswith('GEOMETRY.' + k + '.') and x.find('.' + obj + '.') != -1]
                                if len(configuration_paths) == 0:
                                    errs.append(obj + " in GEOMETRY." + k + ".TIMESERIES.OBJECTS is missing from GEOMETRY." + k)
                        
                    # Must have nites as keys
                    if "NITES" not in self.config_dict['GEOMETRY'][k][config_k].keys():
                        errs.append("GEOMETRY." + k + ".TIMESERIES is missing the NITES parameter")
                    else:
                        if not isinstance(self.config_dict['GEOMETRY'][k][config_k]["NITES"], list):
                            errs.append("GEOMETRY." + k + ".TIMESERIES.NITES must be a list")
                        else:
                            # listed nights must be numeric
                            try:
                                nites = [int(float(x)) for x in self.config_dict['GEOMETRY'][k][config_k]["NITES"]]
                                del nites
                            except TypeError:
                                errs.append("Listed NITES in GEOMETRY." + k + ".TIMESERIES.NITES must be numeric")

                    # Impose restriction on num_exposures
                    if isinstance(self.config["SURVEY"]["PARAMETERS"]["num_exposures"], dict):
                        errs.append("You must set SURVEY.PARAMETERS.num_exposures to 1 if you use TIMESERIES")
                    else:
                        if self.config["SURVEY"]["PARAMETERS"]["num_exposures"] != 1:
                            errs.append("You must set SURVEY.PARAMETERS.num_exposures to 1 if you use TIMESERIES")

                # unexpected entry
                else:
                    errs.append('GEOMETRY.' + k + '.' + config_k + ' is not a valid entry')
    
            # Planes must be indexed sequentially
            if len(detected_planes) != max(detected_planes):
                errs.append("PLANEs in the GEOMETRY." + k + " section must be indexed as 1, 2, 3, ...")

            # Noise sources must be indexed sequentially    
            if len(detected_noise_sources) != max(detected_noise_sources):
                errs.append("NOISE_SOURCEs in the GEOMETRY." + k + " section must be indexed as 1, 2, 3, ...")
                    
                    
        # Configurations must be indexed sequentially
        if len(detected_configurations) != max(detected_configurations):
            errs.append("CONFIGURATIONs in the GEOMETRY section must be indexed as 1, 2, 3, ...")

        # Fractions must sum to a number between 0.0 and 1.0
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

        

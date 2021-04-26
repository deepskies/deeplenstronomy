"""This is an internal class. It identifies mistakes in the
configuration file before dataset generation begins."""

import glob
from inspect import getfullargspec
import os
import sys

from astropy.io import fits
import pandas as pd
from lenstronomy.LensModel import Profiles as LensModelProfiles
from lenstronomy.LightModel import Profiles as LightModelProfiles

# import lenstronomy models into the global scope so dir() can find them
import lenstronomy
lenstronomy_path = lenstronomy.__file__[:-11] # length of __init__.py
light_models = [x.split('/')[-1][0:-3] for x in glob.glob(lenstronomy_path + 'LightModel/Profiles/*.py') if not x.split('/')[-1].startswith('__')]
lens_models = [x.split('/')[-1][0:-3] for x in glob.glob(lenstronomy_path + 'LensModel/Profiles/*.py') if not x.split('/')[-1].startswith('__')]
for model in light_models:
    exec(f'import lenstronomy.LightModel.Profiles.{model} as {model}_light')
for model in lens_models:
    exec(f'import lenstronomy.LensModel.Profiles.{model} as {model}_lens')


from deeplenstronomy.utils import KeyPathDict, read_cadence_file
import deeplenstronomy.distributions as distributions

class ConfigFileError(Exception): pass
class LenstronomyWarning(Exception): pass

class AllChecks():
    """
    Define checks as methods starting with 'check_'
    Methods must return a list of err_message where
    an empty list means success and a nonempty list means failure
    If failure, the err_messages are printed and sys.exit() is called.
    """
    
    def __init__(self, full_dict, config_dict):
        """
        All check methods are run at instantiation.
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

        # set lenstronomy name map
        self.set_lenstronomy_maps()
        self.lenstronomy_valid_models = {"LightModelProfiles": ['GAUSSIAN', 'GAUSSIAN_ELLIPSE', 'ELLIPSOID', 'MULTI_GAUSSIAN', 'MULTI_GAUSSIAN_ELLIPSE',
                                                                'SERSIC', 'SERSIC_ELLIPSE', 'CORE_SERSIC', 'SHAPELETS', 'SHAPELETS_POLAR', 'SHAPELETS_POLAR_EXP',
                                                                'HERNQUIST', 'HERNQUIST_ELLIPSE', 'PJAFFE', 'PJAFFE_ELLIPSE', 'UNIFORM', 'POWER_LAW', 'NIE',
                                                                'CHAMELEON', 'DOUBLE_CHAMELEON', 'TRIPLE_CHAMELEON', 'INTERPOL', 'SLIT_STARLETS', 'SLIT_STARLETS_GEN2'],
                                         "LensModelProfiles": ['SHIFT', 'NIE_POTENTIAL', 'CONST_MAG', 'SHEAR', 'SHEAR_GAMMA_PSI', 'CONVERGENCE', 'FLEXION',
                                                               'FLEXIONFG', 'POINT_MASS', 'SIS', 'SIS_TRUNCATED', 'SIE', 'SPP', 'NIE', 'NIE_SIMPLE', 'CHAMELEON',
                                                               'DOUBLE_CHAMELEON', 'TRIPLE_CHAMELEON', 'SPEP', 'PEMD', 'SPEMD', 'EPL', 'NFW', 'NFW_ELLIPSE',
                                                               'NFW_ELLIPSE_GAUSS_DEC', 'TNFW', 'CNFW', 'CNFW_ELLIPSE', 'CTNFW_GAUSS_DEC', 'NFW_MC', 'SERSIC',
                                                               'SERSIC_ELLIPSE_POTENTIAL', 'SERSIC_ELLIPSE_KAPPA', 'SERSIC_ELLIPSE_GAUSS_DEC', 'PJAFFE',
                                                               'PJAFFE_ELLIPSE', 'HERNQUIST', 'HERNQUIST_ELLIPSE', 'GAUSSIAN', 'GAUSSIAN_KAPPA',
                                                               'GAUSSIAN_ELLIPSE_KAPPA', 'GAUSSIAN_ELLIPSE_POTENTIAL', 'MULTI_GAUSSIAN_KAPPA',
                                                               'MULTI_GAUSSIAN_KAPPA_ELLIPSE', 'INTERPOL', 'INTERPOL_SCALED', 'SHAPELETS_POLAR', 'SHAPELETS_CART',
                                                               'DIPOLE', 'CURVED_ARC', 'ARC_PERT', 'coreBURKERT', 'CORED_DENSITY', 'CORED_DENSITY_2',
                                                               'CORED_DENSITY_MST', 'CORED_DENSITY_2_MST', 'NumericalAlpha', 'MULTIPOLE', 'HESSIAN']}
        
        # find all check functions
        self.checks = [x for x in dir(self) if x.find('check_') != -1]

        # run checks
        total_errs = []
        for check in self.checks:

            err_messages = eval('self.' + check + '()') 
            total_errs += err_messages

        # report errors to user
        if len(total_errs) != 0:
            _kind_output(total_errs)
            raise ConfigFileError

        return

    ### Helper methods
    def set_lenstronomy_maps(self):
         p = {'GAUSSIAN': ".gaussian.Gaussian",
              'GAUSSIAN_ELLIPSE': ".gaussian.GaussianEllipse",
              'ELLIPSOID': ".ellipsoid.Ellipsoid",
              'MULTI_GAUSSIAN': ".gaussian.MultiGaussian",
              'MULTI_GAUSSIAN_ELLIPSE': ".gaussian.MultiGaussianEllipse",
              'SERSIC': ".sersic.Sersic",
              'SERSIC_ELLIPSE': ".sersic.SersicElliptic",
              'CORE_SERSIC': ".sersic.CoreSersic",
              'SHAPELETS': ".shapelets.Shapelets",
              'SHAPELETS_POLAR': ".shapelets_polar.ShapeletsPolar",
              'SHAPELETS_POLAR_EXP': ".shapelets_polar.ShapeletsPolarExp",
              'HERNQUIST': ".hernquist.Hernquist",
              'HERNQUIST_ELLIPSE': ".hernquist.HernquistEllipse",
              'PJAFFE': ".p_jaffe.PJaffe",
              'PJAFFE_ELLIPSE': ".p_jaffe.PJaffe_Ellipse",
              'UNIFORM': ".uniform.Uniform",
              'POWER_LAW': ".power_law.PowerLaw",
              'NIE': ".nie.NIE",
              'CHAMELEON': ".chameleon.Chameleon",
              'DOUBLE_CHAMELEON': ".chameleon.DoubleChameleon",
              'TRIPLE_CHAMELEON': ".chameleon.TripleChameleon",
              'INTERPOL': ".interpolation.Interpol",
              'SLIT_STARLETS': ".starlets.SLIT_Starlets",
              'SLIT_STARLETS_GEN2': ".starlets.SLIT_Starlets"}
         setattr(self, "lenstronomy_light_map", p)

         d = {"SHIFT": ".alpha_shift.Shift",
              "NIE_POTENTIAL": ".nie_potential.NIE_POTENTIAL",
              "CONST_MAG": ".const_mag.ConstMag",
              "SHEAR": ".shear.Shear",
              "SHEAR_GAMMA_PSI": ".shear.ShearGammaPsi",
              "CONVERGENCE": ".convergence.Convergence",
              "FLEXION": ".flexion.Flexion",
              "FLEXIONFG": ".flexionfg.Flexionfg",
              "POINT_MASS": ".point_mass.PointMass",
              "SIS": ".sis.SIS",
              "SIS_TRUNCATED": ".sis_truncate.SIS_truncate",
              "SIE": ".sie.SIE",
              "SPP": ".spp.SPP",
              "NIE": ".nie.NIE",
              "NIE_SIMPLE": ".nie.NIEMajorAxis",
              "CHAMELEON": ".chameleon.Chameleon",
              "DOUBLE_CHAMELEON": ".chameleon.DoubleChameleon",
              "TRIPLE_CHAMELEON": ".chameleon.TripleChameleon",
              "SPEP": ".spep.SPEP",
              "PEMD": ".pemd.PEMD",
              "SPEMD": "spemd.SPEMD",
              "EPL": "epl.EPL",
              "NFW": ".nfw.NFW",
              "NFW_ELLIPSE": ".nfw_ellipse.NFW_ELLIPSE",
              "NFW_ELLIPSE_GAUSS_DEC": ".gauss_decomposition.NFWEllipseGaussDec",
              "TNFW": ".tnfw.TNFW",
              "CNFW": ".cnfw.CNFW",
              "CNFW_ELLIPSE": ".cnfw_ellipse.CNFW_ELLIPSE",
              "CTNFW_GAUSS_DEC": ".gauss_decomposition.CTNFWGaussDec",
              "NFW_MC": ".nfw_mass_concentration.NFWMC",
              "SERSIC": ".sersic.Sersic",
              "SERSIC_ELLIPSE_POTENTIAL": ".sersic_ellipse_potential.SersicEllipse",
              "SERSIC_ELLIPSE_KAPPA": ".sersic_ellipse_kappa.SersicEllipseKappa",
              "SERSIC_ELLIPSE_GAUSS_DEC": ".gauss_decomposition.SersicEllipseGaussDec",
              "PJAFFE": ".p_jaffe.PJaffe",
              "PJAFFE_ELLIPSE": ".p_jaffe_ellipse.PJaffe_Ellipse",
              "HERNQUIST": ".hernquist.Hernquist",
              "HERNQUIST_ELLIPSE": ".hernquist_ellipse.Hernquist_Ellipse",
              "GAUSSIAN": ".gaussian_potential.Gaussian",
              "GAUSSIAN_KAPPA": ".gaussian_kappa.GaussianKappa",
              "GAUSSIAN_ELLIPSE_KAPPA": ".gaussian_ellipse_kappa.GaussianEllipseKappa",
              "GAUSSIAN_ELLIPSE_POTENTIAL": ".gaussian_ellipse_potential.GaussianEllipsePotential",
              "MULTI_GAUSSIAN_KAPPA": ".multi_gaussian_kappa.MultiGaussianKappa",
              "MULTI_GAUSSIAN_KAPPA_ELLIPSE": ".multi_gaussian_kappa.MultiGaussianKappaEllipse",
              "INTERPOL": ".interpol.Interpol",
              "INTERPOL_SCALED": ".interpol.InterpolScaled",
              "SHAPELETS_POLAR": ".shapelet_pot_polar.PolarShapelets",
              "SHAPELETS_CART": ".shapelet_pot_cartesian.CartShapelets",
              "DIPOLE": ".dipole.Dipole",
              "CURVED_ARC": ".curved_arc.CurvedArc",
              "ARC_PERT": ".arc_perturbations.ArcPerturbations",
              "coreBURKERT": ".coreBurkert.CoreBurkert",
              "CORED_DENSITY": ".cored_density.CoredDensity",
              "CORED_DENSITY_2": ".cored_density_2.CoredDensity2",
              "CORED_DENSITY_MST": ".cored_density_mst.CoredDensityMST",
              "CORED_DENSITY_2_MST": ".cored_density_mst.CoredDensityMST",
              "NumericalAlpha": ".numerical_deflections.NumericalAlpha",
              "MULTIPOLE": ".multipole.Multipole",
              "HESSIAN": ".hessian.Hessian"}
         setattr(self, "lenstronomy_lens_map", d)
         return
    
    @staticmethod
    def config_dict_format(*args):
        """
        From a list of parameters, construct the path through the config dictionary
        """
        return "['" + "']['".join(args) + "']"

    def config_lookup(self, lookup_str, full=False):
        """
        From a key path, get the value in the dictionary

        Args:
            lookup_str (str): path of keys through a nested dictionary
            full (bool, optional, default=False): `True for lookup in the `full_dict`, `False` for lookup in the `config_dict`

        Returns:
            The value in the dictionary at the location of the keypath
        """
        if not full:
            return eval("self.config" + lookup_str)
        else:
            return eval("self.full" + lookup_str)
        
    ### Check functions
    def check_top_level_existence(self):
        """
        Check for the DATASET, SURVEY, IMAGE, COSMOLOGY, SPECIES, and GEOMETRY sections
        in the config file
        """
        errs = []
        for name in ['DATASET', 'SURVEY', 'IMAGE', 'COSMOLOGY', 'SPECIES', 'GEOMETRY']:
            if name not in self.full.keys():
                errs.append("Missing {0} section from config file".format(name))
        return errs

    def check_random_seed(self):
        """
        Check whether the passed value for the random seed is valid
        """
        errs = []
        try:
            seed = int(self.config["DATASET"]["PARAMETERS"]["SEED"])
        except KeyError:
            return [] # random seed not specified
        except ValueError:
            errs.append("DATASET.PARAMETERS.SEED was not able to be converted to an integer")

        return errs
            
    def check_low_level_existence(self):
        """
        Check that the DATASET.NAME, DATASET.PARAMETERS.SIZE, COSMOLOGY.PARAMETERS.H0, 
        COSMOLOGY.PARAMETERS.Om0, IMAGE.PARAMETERS.exposure_time, IMAGE.PARAMETERS.numPix, 
        IMAGE.PARAMETERS.pixel_scale, IMAGE.PARAMETERS.psf_type, IMAGE.PARAMETERS.read_noise,
        IMAGE.PARAMETERS.ccd_gain, SURVEY.PARAMETERS.BANDS, SURVEY.PARAMETERS.seeing, 
        SURVEY.PARAMETERS.magnitude_zero_point, SURVEY.PARAMETERS.sky_brightness, and
        SURVEY.PARAMETERS.num_exposures are all present in the config file
        """
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
                config_obj = self.config_lookup(self.config_dict_format(*param.split('.')))
            except KeyError:
                errs.append(param + " is missing from the Config File")

        return errs

    def check_not_allowed_to_be_drawn_from_a_distribution(self):
        """
        Check that parameters that must be fixed in the simulation (DATASET.NAME,
        DATASET.PARAMETERS.SIZE, DATASET.PARAMETERS.OUTDIR, IMAGE.PARAMETERS.numPix,
        COSMOLOGY.PARAMETERS.H0, COSMOLOGY.PARAMETERS.Tcmb, COSMOLOGY.PARAMETERS.Neff, 
        COSMOLOGY.PARAMETERS.m_nu, and COSMOLOGY.PARAMETERS.Ob0) are not being
        drawn from a distribution with the DISTRIBUTION keyword
        """
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
                config_obj = self.config_lookup(self.config_dict_format(*param.split('.')))
            except KeyError:
                # The checked parameter was not in the config dict
                continue
            
            if isinstance(config_obj, dict):
                errs.append(param + " cannot be drawn from a distribution")
        return errs

    def check_for_auxiliary_files(self):
        """
        Check that any auxiliary files specified with the INPUT keyword are
        able to be found
        """
        errs = []
        input_paths = [x for x in self.full_keypaths if x.find("INPUT") != -1]
        input_files = [self.config_lookup(self.config_dict_format(*param.split('.')), full=True) for param in input_paths]
        for filename in input_files:
            if not os.path.exists(filename):
                errs.append("Unable to find auxiliary file: " + filename)
        return errs

    def check_for_valid_distribution_entry(self):
        """
        Check that use of the DISTRIBUTION keyword in the configuration file (1) points
        to a valid distribution and (2) has an entry for each parameter
        """
        errs = []
        distribution_paths = [x for x in self.full_keypaths if x.endswith("DISTRIBUTION")]
        distribution_dicts = [self.config_lookup(self.config_dict_format(*param.split('.'))) for param in distribution_paths]
        for distribution_dict, path in zip(distribution_dicts, distribution_paths):
            # must have name key - return early to not break the remaining parts of this function
            if "NAME" not in distribution_dict.keys():
                errs.append(path + " is missing the NAME key")
                return errs
            else:
                # name must be valid
                if distribution_dict["NAME"] not in dir(distributions):
                    errs.append(path + "." + distribution_dict["NAME"] +  " is not a valid distribution name")
                    return errs

            allowed_params = list(set(getfullargspec(eval("distributions." + distribution_dict["NAME"]))[0]) - set(['bands', 'seed']))
            remaining_params = allowed_params.copy()
            if len(set(allowed_params) - set(["bands"])) != 0:
                # the requested distribution requires parameters so config dict must have parameter key
                if "PARAMETERS" not in distribution_dict.keys():
                    errs.append(path + " is missing the PARAMETERS key")
                else:
                    # if parameters is not a dict, skip
                    if distribution_dict["PARAMETERS"] is None: 
                        continue
                    elif not isinstance(distribution_dict["PARAMETERS"], dict):
                        errs.append(path + '.PARAMETERS must be a dictionary or None')
                    else:
                        # parameters must be valid for the distribution
                        for param in distribution_dict["PARAMETERS"]:
                            if param not in allowed_params:
                                errs.append(path + '.PARAMETERS.' + param + ' is not in the allowed list of ({0}) for the distribtuion '.format(', '.join(allowed_params)) + distribution_dict["NAME"]) 
                            else:
                                remaining_params.pop(remaining_params.index(param))

                        if len(remaining_params) != 0:
                            errs.append(path + ".PARAMETERS is missing parameters: " + ', '.join(remaining_params))
                                
        return errs
    
    def check_input_distributions(self):
        """
        Check that a USERDIST file can be read in and has the proper format
        """
        errs = []
        if "DISTRIBUTIONS" in self.config.keys():
            # there must be at least 1 USERDIST_ key
            userdists = [x for x in self.config["DISTRIBUTIONS"].keys() if x.startswith("USERDIST_")]
            if len(userdists) == 0:
                errs.append("DISTRIBUTIONS section must have at least 1 USERDIST key")
            else:
                for userdist in userdists:
                    # must be a dictionary
                    if not isinstance(self.config["DISTRIBUTIONS"][userdist], dict):
                        errs.append("DISTRIBUTIONS." + userdist + " must be a dictionary with keys FILENAME and MODE")
                    else:
                        # must specify FILENAME and MODE - return early if these are missing to avoid future errors
                        for param in ['FILENAME', 'MODE']:
                            if param not in self.config["DISTRIBUTIONS"][userdist].keys():
                                errs.append("DISTRIBUTIONS." + userdist + " is missing the " + param + " key")
                                return errs
                        
                        # specified file must exist
                        if not os.path.exists(self.config["DISTRIBUTIONS"][userdist]['FILENAME']):
                            errs.append("DISTRIBUTIONS." + userdist + " File '" + self.config["DISTRIBUTIONS"][userdist]['FILENAME'] + "' not found")
                        else:
                            # must be able to read file
                            df = None
                            try:
                                df = pd.read_csv(self.config["DISTRIBUTIONS"][userdist]['FILENAME'], delim_whitespace=True)
                                if "WEIGHT" not in df.columns:
                                    errs.append("WEIGHT column not found in  DISTRIBUTIONS." + userdist + "File '" + self.config["DISTRIBUTIONS"][userdist]['FILENAME'] + "'")
                            except Exception as e:
                                errs.append("Error reading DISTRIBUTIONS." + userdist + " File '" + self.config["DISTRIBUTIONS"][userdist]['FILENAME'] + "'")
                            finally:
                                del df

                        # mode must be valid
                        if self.config["DISTRIBUTIONS"][userdist]['MODE'] not in ['interpolate', 'sample']:
                            errs.append("DISTRIBUTIONS." + userdist + ".MODE must be either 'interpolate' or 'sample'")

                        # if step is specified, it must be an integer
                        if 'STEP' in self.config["DISTRIBUTIONS"][userdist].keys():
                            if not isinstance(self.config["DISTRIBUTIONS"][userdist]['STEP'], int):
                                errs.append("DISTRIBUTIONS." + userdist + ".STEP must be a positive integer")
                            else:
                                if self.config["DISTRIBUTIONS"][userdist]['STEP'] < 1:
                                    errs.append("DISTRIBUTIONS." + userdist + ".STEP must be a positive integer")
        return errs

    def check_image_backgrounds(self):
        """
        Check that images used for backgrounds can be read in and organized successfully
        """
        errs = []
        if "BACKGROUNDS" in self.config.keys():
            # value must be a dict
            if not isinstance(self.config["BACKGROUNDS"], dict):
                errs.append("BACKGROUNDS must be a dict with keys PATH and CONFIGURATIONS")
            else:
                if not "PATH" in self.config["BACKGROUNDS"].keys():
                    errs.append("BACKGROUNDS.PATH is missing from configuration file")
                    return errs
                else:
                    # directory must exist
                    if not os.path.exists(self.config["BACKGROUNDS"]["PATH"]):
                        errs.append("BACKGROUNDS.PATH directory '" + self.config["BACKGROUNDS"]["PATH"] + "' not found")
                    else:
                        
                        dimensions = {}
                        # one file must exist per band
                        for band in self.config["SURVEY"]["PARAMETERS"]["BANDS"].split(','):
                            if not os.path.exists(self.config["BACKGROUNDS"]["PATH"] + "/" + band + ".fits"):
                                errs.append("BACKGROUNDS: " + self.config["BACKGROUNDS"]["PATH"] + '/' + band + ".fits is missing")
                            else:
                                # must be able to open file
                                hdu, data = None, None
                                try:
                                    hdu = fits.open(self.config["BACKGROUNDS"]["PATH"] + '/' + band + '.fits')
                                    data = hdu[0].data
                                    if len(data.shape) != 3:
                                        errs.append("image data in " + self.config["BACKGROUNDS"]["PATH"] + '/' + band + '.fits is formatted incorrectly')
                                    dimensions[band] = data.shape[0]
                                except Exception:
                                    errs.append("Error reading " + self.config["BACKGROUNDS"]["PATH"] + '/' + band + '.fits')
                                finally:
                                    if hdu is not None:
                                        hdu.close()
                                    del data

                        # map.txt file is formatted correctly
                        if os.path.exists(self.config["BACKGROUNDS"]["PATH"] + '/map.txt'):
                            df = None
                            try:
                                df = pd.read_csv(self.config["BACKGROUNDS"]["PATH"] + '/' + 'map.txt', delim_whitespace=True)
                                dimensions["map"] = df.shape[0]
                            except Exception:
                                err.append("Error reading " + self.config["BACKGROUNDS"]["PATH"] + '/map.txt')
                            finally:
                                del df

                        # dimensions of images and (optional) map must be the same
                        if len(set(dimensions.values())) != 1:
                            errs.append("BACKGROUNDS: dimensions of images files and possibly map.txt are inconsistent")


                if not "CONFIGURATIONS" in self.config["BACKGROUNDS"].keys():
                    errs.append("BACKGROUNDS.CONFIGURATIONS is missing from the config file")
                else:
                    # must be a list
                    if not isinstance(self.config["BACKGROUNDS"]["CONFIGURATIONS"], list):
                        errs.append("BACKGROUNDS.CONFIGURATIONS must be a list of configurations like ['CONFIGURATION_1', 'CONFIGURATION_3']")
                    else:
                        # list entries must be strings
                        for entry in self.config["BACKGROUNDS"]["CONFIGURATIONS"]:
                            if not isinstance(entry, str):
                                errs.append("BACKGROUNDS.CONFIGURATIONS list entries must be strings like 'CONFIGURATION_1'")
                            else:
                                # list entries must be names of configurations in the geometry section
                                if entry not in self.config["GEOMETRY"].keys():
                                    errs.append("BACKGROUNDS.CONFIGURATIONS entry {0} is not in the GEOMETRY section".format(entry))

        return errs
    
    def _valid_model(self, model_name, path):
        errs = []

        # check that transmission curves exist for the bands
        if model_name not in ['flat', 'flatnoise', 'variable', 'variablenoise', 'static']:
            if not self.checked_ts_bands:
                for band in self.config["SURVEY"]["PARAMETERS"]["BANDS"].split(','):
                    try:
                        filter_file = [x for x in glob.glob('filters/*_' + band + '.*')][0]
                        passband = pd.read_csv(filter_file,
                                               names=['WAVELENGTH', 'TRANSMISSION'],
                                               delim_whitespace=True, comment='#')
                    except Exception:
                        if band in ['g', 'r', 'i', 'z', 'Y']:
                            print("Warning: Unable to find transmission curve for " + band + " in the filters/ directory")
                            print("\tIf this is the first time using TIMESERIES, the transmission curve will be downloaded automatically")
                        else:
                            errs.append("Unable to find transmission curve for " + band + " in the filters/ directory")
                self.checked_ts_bands = True

            # check that the model name is allowed
            try:
                obj = model_name.split('_')[0]
                sed = model_name.split('_')[1]
            except IndexError:
                errs.append(path + '.' + model_name + ' is formatted incorrectly; use MODEL: <obj>_<sed>')
                obj, sed = 'ia', 'random'

            if obj == 'ia':
                if sed not in ['random', 'salt2-template-0.dat', 'snflux-1a-Nugent2002.dat']:
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
            elif obj == 'kn':
                pass    
            elif obj == 'user':
                if not os.path.exists("seds/user/" + sed):
                    errs.append(path + '.' + model_name + ' sed file ({0}) is missing'.format(sed))
                else:
                    # check that the file can be opened properly
                    try:
                        df = pd.read_csv('seds/user/' + sed,
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
        if "NAME" not in self.config['SPECIES'][k].keys():
            errs.append("SPECIES." + k + " is missing an entry for NAME")
        else:
            # name must be a string
            if not isinstance(self.config['SPECIES'][k]["NAME"], str):
                errs.append("SPECIES." + k + ".NAME must be the name of a function in distribution.py")
            else:
                names.append(self.config['SPECIES'][k]["NAME"])

        # Check LIGHT_PROFILEs, MASS_PROFILEs, and SHEAR_PROFILEs
        detected_light_profiles, detected_mass_profiles, detected_shear_profiles = [], [], []
        for profile_k in self.config['SPECIES'][k].keys():
            if profile_k.startswith('LIGHT_PROFILE_') or profile_k.startswith('MASS_PROFILE_') or profile_k.startswith('SHEAR_PROFILE_'):
                #set profile_type
                if profile_k.startswith('LIGHT_PROFILE_'):
                    profile_type = "LightModelProfiles"
                    lenstronomy_map = self.lenstronomy_light_map
                else:
                    profile_type = "LensModelProfiles"
                    lenstronomy_map = self.lenstronomy_lens_map
                
                # Index must be valid
                detections, errors = self._valid_index(profile_k, "SPECIES." + k)
                if profile_k.startswith('LIGHT_PROFILE_'):
                    detected_light_profiles += detections
                elif profile_k.startswith('MASS_PROFILE_'):
                    detected_mass_profiles += detections
                elif profile_k.startswith('SHEAR_PROFILE_'):
                    detected_shear_profiles += detections
                errs += errors

                # Must have name - return early if no name exists
                if "NAME" not in self.config['SPECIES'][k][profile_k].keys():
                    errs.append("SPECIES." + k + "." + profile_k + " needs a NAME")
                else:
                    if not isinstance(self.config['SPECIES'][k][profile_k]["NAME"], str):
                        errs.append("SPECIES." + k + "." + profile_k + ".NAME must be a single name")
                        return errs
                    else:
                        # name must be a valid lenstronomy profile
                        if self.config['SPECIES'][k][profile_k]["NAME"] not in self.lenstronomy_valid_models[profile_type]:
                            errs.append("SPECIES." + k + "." + profile_k + " (" + self.config['SPECIES'][k][profile_k]["NAME"] + ") is not a valid lenstronomy profile")
                        elif lenstronomy_map[self.config['SPECIES'][k][profile_k]["NAME"]] == "warn":
                            # warn about unstable / incompatible profiles
                            errs.append("The lenstronomy model " + self.config['SPECIES'][k][profile_k]["NAME"] + " is not usable within deeplenstronomy")
                # Must have parameters
                if "PARAMETERS" not in self.config['SPECIES'][k][profile_k].keys():
                    errs.append("SPECIES." + k + "." + profile_k + " needs PARAMETERS")
                else:
                    if not isinstance(self.config['SPECIES'][k][profile_k]["PARAMETERS"], dict):
                        errs.append("SPECIES." + k + "." + profile_k + ".PARAMETERS must contain all parameters for the lenstronomy profile")
                    else:
                        # specified parameters must be what lenstronomy is expecting
                        for param_name in self.config['SPECIES'][k][profile_k]["PARAMETERS"].keys():
                            if param_name not in getfullargspec(eval(profile_type + lenstronomy_map[self.config['SPECIES'][k][profile_k]["NAME"]] + ".function"))[0]:
                                if param_name not in ['magnitude', 'sigma_v']:
                                    #lenstronomy functions use `amp` but deeplenstronomy works with `magnitude`
                                    #allow sigma_v to be used as a way to parameterize the lensing
                                    errs.append("SPECIES." + k + "." + profile_k + ".PARAMETERS." + param_name + " is not a valid_parameter for " + self.config['SPECIES'][k][profile_k]["NAME"])
                        
            # If MODEL is specified, it must be valid
            if profile_k == "MODEL":
                if not isinstance(self.config['SPECIES'][k][profile_k], str):
                    errs.append("SPECIES." + k + "." + profile_k + ".MODEL must be a single name")
                else:
                    errs += self._valid_model(self.config['SPECIES'][k][profile_k], "SPECIES." + k + "." + profile_k)

        # need at least one light profile
        if len(detected_light_profiles) < 1:
            errs.append("SPECIES." + k + " needs at least one LIGHT_PROFILE")
        # need at least one mass profile
        if len(detected_mass_profiles) < 1:
            errs.append("SPECIES." + k + " needs at least one MASS_PROFILE (this is a new requirement as of version 0.0.1.8)")
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
        if "NAME" not in self.config['SPECIES'][k].keys():
            errs.append("SPECIES." + k + " is missing an entry for NAME")
        else:
            # name must be a string
            if not isinstance(self.config['SPECIES'][k]["NAME"], str):
                errs.append("SPECIES." + k + ".NAME must be a sinlge unique value")
            else:
                names.append(self.config['SPECIES'][k]["NAME"])

        # Must have a host key
        if "HOST" not in self.config['SPECIES'][k].keys():
            errs.append("SPECIES." + k + " must have a valid HOST")
        else:
            # host name must be a single value
            if not isinstance(self.config['SPECIES'][k]["HOST"], str):
                errs.append("SPECIES." + k + ".HOST must be a single name")
            elif self.config['SPECIES'][k]["HOST"] == "Foreground":
                pass
            else:
                # host must appear in SPECIES section
                species_paths = [self.config_lookup(self.config_dict_format(*x.split('.'))) for x in self.config_keypaths if x.startswith("SPECIES.") and x.endswith(".NAME")]
                species_paths = [x for x in species_paths if x == self.config['SPECIES'][k]["HOST"]]
                if len(species_paths) == 0:
                    errs.append("HOST for SPECIES." + k + " is not found in SPECIES section")

        # Must have PARAMETERS
        if "PARAMETERS" not in self.config['SPECIES'][k].keys():
            errs.append("SPECIES." + k + " must have PARAMETERS")
        else:
            if not isinstance(self.config['SPECIES'][k]["PARAMETERS"], dict):
                errs.append("SPECIES." + k + ".PARAMETERS must be a dictionary")
            else:
                # separation must be used properly
                if "sep" in self.config['SPECIES'][k]["PARAMETERS"].keys():
                    # sep unit must be specified
                    if "sep_unit" not in self.config['SPECIES'][k]["PARAMETERS"].keys():
                        errs.append("sep is specified for SPECIES." + k + ".PARAMETERS but sep_unit is missing")
                    else:
                        if not isinstance(self.config['SPECIES'][k]["PARAMETERS"]["sep_unit"], str):
                            errs.append("SPECIES." + k + ".PARAMETERS.sep_unit must be either 'arcsec' or 'kpc'")
                        else:
                            if self.config['SPECIES'][k]["PARAMETERS"]["sep_unit"] not in ['arcsec', 'kpc']:
                                errs.append("SPECIES." + k + ".PARAMETERS.sep_unit must be either 'arcsec' or 'kpc'")

                # magnitude must be one of the parameters
                if "magnitude" not in self.config['SPECIES'][k]["PARAMETERS"].keys():
                    errs.append("SPECIES." + k + ".PARAMETERS.magnitude must be specified")

        # If timeseries model is specified, it must be a valid model
        if "MODEL" in self.config['SPECIES'][k].keys():
            if not isinstance(self.config['SPECIES'][k]["MODEL"], str):
                errs.append("SPECIES." + k + ".MODEL must be a single name")
            else:
                errs += self._valid_model(self.config['SPECIES'][k]["MODEL"], "SPECIES." + k + '.MODEL')
                    
        return errs, names

    def _valid_noise(self, k):
        errs, names = [], []
        # Must have name key
        if "NAME" not in self.config['SPECIES'][k].keys():
            errs.append("SPECIES." + k + " is missing an entry for NAME")
        else:
            # name must be a string 
            if not isinstance(self.config['SPECIES'][k]["NAME"], str):
                errs.append("SPECIES." + k + ".NAME must be the name of a function in distribution.py")
            else:
                names.append(self.config['SPECIES'][k]["NAME"])

            # name must be a valid distribution
            if self.config['SPECIES'][k]["NAME"].lower() not in dir(distributions):
                errs.append("SPECIES." + k + ".NAME must be the name of a function in distribution.py")

        # Must have parameter key
        if "PARAMETERS" not in self.config['SPECIES'][k].keys():
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
        """
        Check that all GALAXY, POINTSOURCE, and NOISE objects are formatted correctly
        """
        errs, names = [], []

        # There must be at least one species
        if len(list(self.config['SPECIES'].keys())) == 0:
            errs.append("SPECIES sections needs at least one SPECIES")

        # Check keys
        detected_galaxies, detected_point_sources, detected_noise_sources = [], [], []
        for k in self.config['SPECIES'].keys():
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
        if len(detected_galaxies) !=0 and len(detected_galaxies) != max(detected_galaxies):
            errs.append('GALAXY objects in SPECIES must be indexed like 1, 2, 3, ...')
        if len(detected_point_sources) != 0 and len(detected_point_sources) != max(detected_point_sources):
            errs.append('POINTSOURCE objects in SPECIES must be indexed like 1, 2, 3, ...')
        if len(detected_noise_sources) != 0 and len(detected_noise_sources) != max(detected_noise_sources):
            errs.append('NOISE objects in SPECIES must be indexed like 1, 2, 3, ...')

        # All objects must have a unique name
        if len(set(names)) != len(names):
            errs.append("All entries in SPECIES must have a unique NAME")

        return errs
    
    def check_valid_geometry(self):
        """
        Check that all configurations in the geometry section are formatted correctly
        """
        errs = []

        # There must be at least one configuration
        if len(list(self.config['GEOMETRY'].keys())) == 0:
            errs.append("GEOMETRY sections needs at least one CONFIGURATION")
        
        # Check keys
        detected_configurations, fractions = [], []
        for k in self.config['GEOMETRY'].keys():
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
            if "FRACTION" not in self.config['GEOMETRY'][k].keys():
                errs.append("GEOMETRY." + k + " .FRACTION is missing")
            else:
                try:
                    fraction = float(self.config['GEOMETRY'][k]['FRACTION'])
                    fractions.append(fraction)
                except TypeError:
                    errs.append("GEOMETRY." + k + " .FRACTION must be a float")

            # Configurations must have information
            if len(list(self.config['GEOMETRY'][k].keys())) == 0:
                errs.append("GEOMETRY." + k + " is empty")

            detected_planes, detected_noise_sources = [], []
            for config_k in self.config['GEOMETRY'][k].keys():
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
                    try:
                        if 'REDSHIFT' not in self.config['GEOMETRY'][k][config_k]['PARAMETERS'].keys():
                            errs.append('REDSHIFT is missing from GEOMETRY.' + k + '.' + config_k)
                    except AttributeError:
                        errs.append('Incorrect format detected in ' + k + '.' + config_k)
                        
                    detected_objects = []
                    for obj_k in self.config['GEOMETRY'][k][config_k].keys():
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
                            if not isinstance(self.config['GEOMETRY'][k][config_k][obj_k], str):
                                errs.append('GEOMETRY.' + k + '.' + config_k + '.' + obj_k + ' must be a single name')

                            species_paths = [self.config_lookup(self.config_dict_format(*x.split('.'))) for x in self.config_keypaths if x.startswith('SPECIES.') and x.endswith('.NAME')]
                            species_paths = [x for x in species_paths if x == self.config['GEOMETRY'][k][config_k][obj_k]]
                            if len(species_paths) == 0:
                                errs.append('GEOMETRY.' + k + '.' + config_k + '.' + obj_k + '(' + self.config['GEOMETRY'][k][config_k][obj_k] + ') is missing from the SPECIES section')
                                
                    # Objects must be indexed sequentially
                    if len(detected_objects) != max(detected_objects):
                        errs.append("OBJECTs in the GEOMETRY." + k + '.' + config_k + " section must be indexed as 1, 2, 3, ...")

                # check noise properties
                elif config_k.startswith('NOISE_SOURCE_'):
                    # index must be a valid integer
                    try:
                        val = int(config_k.split('_')[-1])
                        if val < 1:
                            errs.append('GEOMETRY.' + k + '.' + config_k + ' is an invalid Config File entry')
                        detected_noise_sources.append(val)
                    except TypeError:
                        errs.append('GEOMETRY.' + k + '.' + config_k + ' needs a valid integer index greater than zero')

                    # Noise sources must have a single value that appears in the species section
                    if not isinstance(self.config['GEOMETRY'][k][config_k], str):
                        errs.append('GEOMETRY.' + k + '.' + config_k + ' must be a single name')

                    species_paths = [self.config_lookup(self.config_dict_format(*x.split('.'))) for x in self.config_keypaths if x.startswith('SPECIES.') and x.endswith('.NAME')]
                    species_paths = [x for x in species_paths if x == self.config['GEOMETRY'][k][config_k]]
                    if len(species_paths) == 0:
                        errs.append('GEOMETRY.' + k + '.' + config_k + ' is missing from the SPECIES section')
                        
                # check timeseries properties
                elif config_k == 'TIMESERIES':
                    # Must have objects as keys
                    if "OBJECTS" not in self.config['GEOMETRY'][k][config_k].keys():
                        errs.append("GEOMETRY." + k + ".TIMESERIES is missing the OBJECTS parameter")
                    else:
                        if not isinstance(self.config['GEOMETRY'][k][config_k]["OBJECTS"], list):
                            errs.append("GEOMETRY." + k + ".TIMESERIES.OBJECTS must be a list")
                        else:
                            # listed objects must appear in species section, in the configuration, and have a model defined
                            for obj in self.config['GEOMETRY'][k][config_k]['OBJECTS']:
                                species_paths = [x for x in self.config_keypaths if x.startswith('SPECIES.') and x.endswith('.NAME')]
                                species_paths = ['.'.join(x.split('.')[:-1]) for x in species_paths if self.config_lookup(self.config_dict_format(*x.split('.'))) == obj]
                                if len(species_paths) == 0:
                                    errs.append(obj + " in GEOMETRY." + k + ".TIMESERIES.OBJECTS is missing from the SPECIES section")
                                elif "MODEL" not in self.config_lookup(self.config_dict_format(*species_paths[0].split('.'))).keys():
                                    errs.append("MODEL for " + obj + " in GEOMETRY." + k + ".TIMESERIES.OBJECTS is missing from the SPECIES section")
                                configuration_paths = [x for x in self.config_keypaths if x.startswith('GEOMETRY.' + k + '.') and x.find('.OBJECT_') != -1]
                                configuration_paths = [x for x in configuration_paths if self.config_lookup(self.config_dict_format(*x.split('.'))) == obj]
                                if len(configuration_paths) == 0:
                                    errs.append(obj + " in GEOMETRY." + k + ".TIMESERIES.OBJECTS is missing from GEOMETRY." + k)
                        
                    # Must have nites as keys
                    if "NITES" not in self.config['GEOMETRY'][k][config_k].keys():
                        errs.append("GEOMETRY." + k + ".TIMESERIES is missing the NITES parameter")
                    else:
                        if not (isinstance(self.config['GEOMETRY'][k][config_k]["NITES"], list) or isinstance(self.config['GEOMETRY'][k][config_k]["NITES"], str)):
                            errs.append("GEOMETRY." + k + ".TIMESERIES.NITES must be a list or a filename")
                        else:
                            if isinstance(self.config['GEOMETRY'][k][config_k]["NITES"], list):
                                nitelists = [self.config['GEOMETRY'][k][config_k]["NITES"]]
                            else:
                                # filename of cadence file
                                try:
                                    cadence_dict = read_cadence_file(self.config['GEOMETRY'][k][config_k]["NITES"])

                                    # Pointings must be incrementally sequenced
                                    nitelists = []
                                    bands = set(self.config['SURVEY']['PARAMETERS']['BANDS'].strip().split(','))
                                    pointings = [x for x in cadence_dict.keys() if x.startswith('POINTING_')]
                                    if len(pointings) == 0:
                                        errs.append("GEOMETRY." + k + ".TIMESERIES.NITES." + self.config['GEOMETRY'][k][config_k]["NITES"] + " contains no POINTING entries")
                                    for pointing in pointings:
                                        if set(list(cadence_dict[pointing].keys())) != bands:
                                            errs.append("GEOMETRY." + k + ".TIMESERIES.NITES." + self.config['GEOMETRY'][k][config_k]["NITES"] + pointing + " does not contain same bands as the survey")
                                        else:
                                            cad_length = len(cadence_dict[pointing][self.config['SURVEY']['PARAMETERS']['BANDS'].strip().split(',')[0]])
                                            for band in bands:
                                                if len(cadence_dict[pointing][band]) != cad_length:
                                                    errs.append("GEOMETRY." + k + ".TIMESERIES.NITES." + self.config['GEOMETRY'][k][config_k]["NITES"] + pointing + " contains cadences of different lengths")
                                                nitelists.append(cadence_dict[pointing][band])
                                    
                                except Exception:
                                    errs.append("GEOMETRY." + k + ".TIMESERIES.NITES." + self.config['GEOMETRY'][k][config_k]["NITES"] + " caused an error when reading file")
                                    nitelists = [[]]
                                    
                            for nitelist in nitelists:
                                # listed nights must be numeric
                                try:
                                    nites = [int(float(x)) for x in nitelist]
                                    del nites
                                except TypeError:
                                    errs.append("Listed NITES in GEOMETRY." + k + ".TIMESERIES.NITES must be numeric")

                    # Check validity of PEAK argument, if passed
                    if "PEAK" in self.config['GEOMETRY'][k][config_k].keys():
                        if not isinstance(self.config['GEOMETRY'][k][config_k]["PEAK"], dict):
                            try:
                                peak = int(float(self.config['GEOMETRY'][k][config_k]["PEAK"]))
                                del peak
                            except TypeError:
                                errs.append("PEAK argument in GEOMETRY." + k + ".TIMESERIES.PEAK must be numeric")
                                
                    # Impose restriction on num_exposures
                    if isinstance(self.config["SURVEY"]["PARAMETERS"]["num_exposures"], dict):
                        errs.append("You must set SURVEY.PARAMETERS.num_exposures to 1 if you use TIMESERIES")
                    else:
                        if self.config["SURVEY"]["PARAMETERS"]["num_exposures"] < 0.99 or self.config["SURVEY"]["PARAMETERS"]["num_exposures"] > 1.01:
                            errs.append("You must set SURVEY.PARAMETERS.num_exposures to 1 if you use TIMESERIES")

                elif config_k == 'NAME' or config_k == 'FRACTION':
                    pass
                
                # unexpected entry
                else:
                    errs.append('GEOMETRY.' + k + '.' + config_k + ' is not a valid entry')
    
            # Planes must be indexed sequentially
            if len(detected_planes) != max(detected_planes):
                errs.append("PLANEs in the GEOMETRY." + k + " section must be indexed as 1, 2, 3, ...")

            # Must have at least 2 planes
            if len(detected_planes) < 2:
                errs.append("GEOMETRY." + k + " must have at least 2 planes (this is a new requirement as of version 0.0.1.8)")

            # Noise sources must be indexed sequentially
            if len(detected_noise_sources) != 0 and len(detected_noise_sources) != max(detected_noise_sources):
                errs.append("NOISE_SOURCEs in the GEOMETRY." + k + " section must be indexed as 1, 2, 3, ...")
                    
                    
        # Configurations must be indexed sequentially
        if len(detected_configurations) != max(detected_configurations):
            errs.append("CONFIGURATIONs in the GEOMETRY section must be indexed as 1, 2, 3, ...")

        # Fractions must sum to a number between 0.0 and 1.0
        if not (0.0 < sum(fractions) <= 1.0):
            errs.append("CONFIGURATION FRACTIONs must sum to a number between 0.0 and 1.0")
                
        return errs
    
    # End check functions

def _kind_output(errs):
    """
    Print all detected errors in the configuration file to the screen

    Args:
        errs (List[str]): A list of error messages as strings
    """
    for err in errs:
        print(err)
    return


def _run_checks(full_dict, config_dict):
    """
    Instantiate an AllChecks object to run checks

    Args:
        full_dict (dict): a Parser.full_dict object 
        config_dict (dict): a Parser.config_dict object 
    """
    try:
        check_runner = AllChecks(full_dict, config_dict)
    except ConfigFileError:
        print("\nFatal error(s) detected in config file. Please edit and rerun.")
        raise ConfigFileError
        
    return

        

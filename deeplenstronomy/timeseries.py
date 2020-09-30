# Class for light curve generation from time-series spectral energy distributions

import glob
import os
import random
import warnings
warnings.filterwarnings("ignore")

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import quad

class LCGen():
    """Light Curve Generation"""    
    def __init__(self, bands=''):
        """
        Initialize the LCGen object
        
        :param bands: comma-separated string of bands, i.e. 'g,r,i,z'
        """
        # Check for filtes / seds and download if necessary
        self.__download_data()
        
        # Collect sed files
        self.ia_sed_files = glob.glob('seds/ia/*.dat')
        self.cc_sed_files = glob.glob('seds/cc/*.SED')
        self.__read_cc_weights()
        
        # Collect filter transmission curves
        self.filter_files = glob.glob('filters/*.dat')
        self.bands = bands.split(',')
        
        # Interpolate the transmission curves
        for band in self.bands:
            transmission_frequency, transmission_wavelength = self.__read_passband(band)
            setattr(self, '{0}_transmission_frequency'.format(band), transmission_frequency)
            setattr(self, '{0}_transmission_wavelength'.format(band), transmission_wavelength)
        
        return

    def __download_data(self):
        """
        Check for required data and download if it is missing
        """
        if not os.path.exists('seds'):
            os.mkdir('seds')
        if not os.path.exists('seds/ia'):
            os.system('svn checkout https://github.com/rmorgan10/deeplenstronomy_data/trunk/seds/ia')
            os.system('mv ia seds')
        if not os.path.exists('seds/cc'):
            os.system('svn checkout https://github.com/rmorgan10/deeplenstronomy_data/trunk/seds/cc')
            os.system('mv cc seds')
        if not os.path.exists('seds/kn'):
            os.system('svn checkout https://github.com/rmorgan10/deeplenstronomy_data/trunk/seds/kn')
            os.system('mv kn seds')
        if not os.path.exists('filters'):
            os.system('svn checkout https://github.com/rmorgan10/deeplenstronomy_data/trunk/filters')
    
    def __read_cc_weights(self):
        """
        Read Core-Collapse SNe metadata
        
        :assign cc_info_df: dataframe containing all CC template metadata
        :assign cc_weights: lists of weights for each template
        """
        df = pd.read_csv('seds/cc/SIMGEN_INCLUDE_NON1A.INPUT', comment='#', delim_whitespace=True)
        self.cc_info_df = df
        self.cc_weights = [df['WGT'].values[df['SED'].values == x.split('/')[-1].split('.')[0]][0] for x in self.cc_sed_files]
        return
        
    
    def __read_passband(self, band):
        """
        Read and interolate filter transmission curves
        
        :param band: the single-letter band identifier
        :return: transmisison_frequency: interpolated filter transmission as a function of frequency
        :return: transmisison_wavelength: interpolated filter transmission as a function of wavelength
        """
        #Target filter file associated with band
        filter_file = [x for x in self.filter_files if x.find('_' + band) != -1][0]
        
        # Read and format filter transmission info
        passband = pd.read_csv(filter_file, 
                               names=['WAVELENGTH', 'TRANSMISSION'], 
                               delim_whitespace=True, comment='#')
        setattr(self, '_{0}_obs_frame_freq_min'.format(band), 2.99792458e18 / np.max(passband['WAVELENGTH'].values))
        setattr(self, '_{0}_obs_frame_freq_max'.format(band), 2.99792458e18 / np.min(passband['WAVELENGTH'].values))
        
        # Add boundary terms to cover the whole range
        passband.loc[passband.shape[0]] = (1.e-1, 0.0)
        passband.loc[passband.shape[0]] = (4.e+4, 0.0)
        
        # Convert to frequency using speed of light in angstroms
        passband['FREQUENCY'] = 2.99792458e18 / passband['WAVELENGTH'].values
        setattr(self, '{0}_obs_frame_transmission'.format(band), passband)
        
        # Interpolate and return
        transmission_frequency = interp1d(passband['FREQUENCY'].values, passband['TRANSMISSION'].values, fill_value=0.0)
        transmission_wavelength = interp1d(passband['WAVELENGTH'].values, passband['TRANSMISSION'].values, fill_value=0.0)
        return transmission_frequency, transmission_wavelength

    
    def _read_sed(self, sed_filename):
        """
        Read a Spectral Enerrgy Distribution into a dataframe
        
        :param sed_filename: name of file describing the sed
        :return: sed: a dataframe of the sed
        """
        
        # Read and format sed info
        sed = pd.read_csv(sed_filename,
                          names=['NITE', 'WAVELENGTH_REST', 'FLUX'], 
                          delim_whitespace=True, comment='#')
        for nite in np.unique(sed['NITE'].values):
            sed.loc[sed.shape[0]] = (nite, 10.0, 0.0)
            sed.loc[sed.shape[0]] = (nite, 25000.0, 0.0)
        sed['FREQUENCY_REST'] = 2.99792458e18 / sed['WAVELENGTH_REST'].values
        
        # Normalize
        func = interp1d(sed['WAVELENGTH_REST'].values, sed['FLUX'].values)
        sed['FLUX'] = sed['FLUX'].values / quad(func, 10.0, 25000.0)[0]
        
        # Round nights to nearest int
        sed['NITE'] = [round(x) for x in sed['NITE'].values]
        
        return sed
    
    def _get_kcorrect(self, sed, band, redshift):
        """
        Calculate the K-Correction
        
        :param sed: the sed on the night of peak flux
        :param band: the single-letter band being used
        :param redshift: the redshift of the object
        :return: kcor: the k-correction to the absolute magnitude
        """
        return -2.5 * np.log10((1.0 + redshift) * 
                               (self._integrate_through_band(sed, band, redshift, frame='OBS') /
                                self._integrate_through_band(sed, band, redshift, frame='REST')))
        
    def _get_distance_modulus(self, redshift, cosmo):
        """
        Calculate the dimming effect of distance to the source
        
        :param redshift: the redshift of the object
        :param cosmo: an astropy.cosmology instance
        :return: dmod: the distance modulus contribution to the apparent magnitude
        """
        return 5.0 * np.log10(cosmo.luminosity_distance(redshift).value * 10 ** 6 / 10)

    
    def _integrate_through_band(self, sed, band, redshift, frame='REST'):
        """
        Calculate the flux through a given band by integrating in frequency
        
        :param sed: a dataframe containing the sed of the object
        :param band: the single-letter filter being used
        :param redshift: the redshift of the source
        :param frame: chose from ['REST', 'OBS'] to choose the rest frame or the observer frame
        :return: flux: the measured flux from the source through the filter
        """
        # Tighten bounds for accurate integration
        lower_bound = eval('self._{0}_obs_frame_freq_min'.format(band))
        upper_bound = eval('self._{0}_obs_frame_freq_max'.format(band))        
        frequency_arr = np.linspace(lower_bound, upper_bound, 100000)
        
        # Make an interpolated version of the integrand
        interpolated_sed = interp1d(sed['FREQUENCY_{0}'.format(frame)].values, sed['FLUX'].values, fill_value=0.0)
        integrand = eval('self.{0}_transmission_frequency(frequency_arr) * interpolated_sed(frequency_arr) / frequency_arr'.format(band))
        interpolated_integrand = interp1d(frequency_arr, integrand, fill_value=0.0)
               
        # Integrate and return
        return quad(interpolated_integrand, lower_bound, upper_bound, limit=500)[0]
    
    def _get_closest_nite(self, unique_nites, nite):
        """
        Return the nite in the sed closest to a desired nite
        
        :param unique_nites: a set of the nights in an sed
        :param nite: the nite you wish to find the closest neighbor for
        :return: closest_nite: the closest nite in the sed to the given nite
        """
        return unique_nites[np.argmin(np.abs(nite - unique_nites))]

    def gen_variable(self, redshift, nites, sed=None, sed_filename=None, cosmo=None):
        """
        Generate a random variable light curve

        :param redshift (ignored)  
        :param nites: a list of night relative to peak you want to obtain a magnitude for 
        :param sed_filename: (ignored)  
        :param cosmo: (ignored)   
        :return: lc_dict: a dictionary with keys ['lc, 'obj_type', 'sed']
            - 'lc' contains a dataframe of the light from the object     
            - 'obj_type' contains a string for the type of object. Will always be 'Variable' here
            - 'sed' contains the filename of the sed used. Will always be 'Variable' here  
        """
        output_data_cols = ['NITE', 'BAND', 'MAG']
        output_data = []
        central_mag = random.uniform(12.0, 23.0)
        colors = {band: mag for band, mag in zip(self.bands, np.random.uniform(low=-2.0, high=2.0, size=len(self.bands)))}
        for nite in nites:
            central_mag = random.uniform(central_mag - 1.0, central_mag + 1.0)
            for band in self.bands:
                output_data.append([nite, band, central_mag + colors[band]])

        return {'lc': pd.DataFrame(data=output_data, columns=output_data_cols),
		'obj_type': 'Variable',
                'sed': 'Variable'}
    
    def gen_flat(self, redshift, nites, sed=None, sed_filename=None, cosmo=None):
        """
        Generate a random flat light curve
        
        :param redshift (ignored)
        :param nites: a list of night relative to peak you want to obtain a magnitude for
        :param sed_filename: (ignored) 
        :param cosmo: (ignored)  
        :return: lc_dict: a dictionary with keys ['lc, 'obj_type', 'sed'] 
            - 'lc' contains a dataframe of the light from the object 
            - 'obj_type' contains a string for the type of object. Will always be 'Flat' here 
            - 'sed' contains the filename of the sed used. Will always be 'Flat' here            
        """
        output_data_cols = ['NITE', 'BAND', 'MAG']
        central_mag = random.uniform(12.0, 23.0)
        mags = {band: mag for band, mag in zip(self.bands, central_mag + np.random.uniform(low=-2.0, high=2.0, size=len(self.bands)))}
        output_data = []
        for nite in nites:
            for band in self.bands:
                output_data.append([nite, band, mags[band]])

        return {'lc': pd.DataFrame(data=output_data, columns=output_data_cols),
                'obj_type': 'Flat',
                'sed': 'Flat'}
    
    def gen_variablenoise(self, redshift, nites, sed=None, sed_filename=None, cosmo=None):
        """ 
        Generate a variable light curve with small random noise
        
        :param redshift (ignored)
        :param nites: a list of night relative to peak you want to obtain a magnitude for 
        :param sed_filename: (ignored) 
        :param cosmo: (ignored) 
        :return: lc_dict: a dictionary with keys ['lc, 'obj_type', 'sed']
            - 'lc' contains a dataframe of the light from the object  
            - 'obj_type' contains a string for the type of object. Will always be 'VariableNoise' here   
            - 'sed' contains the filename of the sed used. Will always be 'VariableNoise' here       
        """
        noiseless_lc_dict = self.gen_variable(redshift, nites)
        noise = np.random.normal(loc=0, scale=0.25, size=noiseless_lc_dict['lc'].shape[0])
        noiseless_lc_dict['lc']['MAG'] = noiseless_lc_dict['lc']['MAG'].values + noise
        noiseless_lc_dict['obj_type'] = 'VariableNoise'
        noiseless_lc_dict['sed'] = 'VariableNoise'
        return noiseless_lc_dict

    
    def gen_flatnoise(self, redshift, nites, sed=None, sed_filename=None, cosmo=None):
        """
        Generate a flat light curve will small random noise

        :param redshift (ignored)
        :param nites: a list of night relative to peak you want to obtain a magnitude for 
        :param sed_filename: (ignored) 
        :param cosmo: (ignored)  
        :return: lc_dict: a dictionary with keys ['lc, 'obj_type', 'sed'] 
            - 'lc' contains a dataframe of the light from the object  
            - 'obj_type' contains a string for the type of object. Will always be 'FlatNoise' here
            - 'sed' contains the filename of the sed used. Will always be 'FlatNoise' here     
        """
        noiseless_lc_dict = self.gen_flat(redshift, nites)
        noise = np.random.normal(loc=0, scale=0.25, size=noiseless_lc_dict['lc'].shape[0])
        noiseless_lc_dict['lc']['MAG'] = noiseless_lc_dict['lc']['MAG'].values + noise
        noiseless_lc_dict['obj_type'] = 'FlatNoise'
        noiseless_lc_dict['sed'] = 'FlatNoise'
        return noiseless_lc_dict
        
    def gen_user(self, redshift, nites, sed=None, sed_filename=None, cosmo=None):
        """
        Generate a light curve from a user-specidied SED

        :param redshfit: the redshift of the source 
        :param nites: a list of night relative to peak you want to obtain a magnitude for  
        :param sed: (optional) a dataframe containing the sed of the SN
        :param cosmo: (optional) an astropy.cosmology instance 
        :return: lc_dict: a dictionary with keys ['lc, 'obj_type', 'sed']
            - 'lc' contains a dataframe of the light from the object
            - 'obj_type' contains a string for the type of object. Will always be <sed_filename> here 
            - 'sed' contains the filename of the sed used  
        """
        if not sed:
            sed = self._read_sed('seds/user/' + sed_filename)

        return self.gen_lc_from_sed(redshift, nites, sed, sed_filename, sed_filename, cosmo=cosmo)

    def gen_kn(self, redshift, nites, sed=None, sed_filename=None, cosmo=None):
        """
        Generate a GW170817-like light curve

        :param redshfit: the redshift of the source   
        :param nites: a list of night relative to peak you want to obtain a magnitude for  
        :param sed: (optional) a dataframe containing the sed of the SN 
        :param cosmo: (optional) an astropy.cosmology instance       
        :return: lc_dict: a dictionary with keys ['lc, 'obj_type', 'sed'] 
            - 'lc' contains a dataframe of the light from the object
            - 'obj_type' contains a string for the type of object. Will always be 'KN' here 
            - 'sed' contains the filename of the sed used  
        """

        sed_filename = 'seds/kn/kn.SED'
        if not sed:
            sed = self._read_sed(sed_filename)

        return self.gen_lc_from_sed(redshift, nites, sed, 'KN', sed_filename, cosmo=cosmo)
    
    def gen_ia(self, redshift, nites, sed=None, sed_filename=None, cosmo=None):
        """
        Generate a SN-Ia light curve
        
        :param redshfit: the redshift of the source
        :param nites: a list of night relative to peak you want to obtain a magnitude for
        :param sed: (optional) a dataframe containing the sed of the SN
        :param sed_filename: (optional) the filename of the sed of the SN
        :param cosmo: (optional) an astropy.cosmology instance
        :return: lc_dict: a dictionary with keys ['lc, 'obj_type', 'sed']
            - 'lc' contains a dataframe of the light from the object
            - 'obj_type' contains a string for the type of object. Will always be 'Ia' here
            - 'sed' contains the filename of the sed used
        """
        
        # Read rest-frame sed if not supplied as argument
        if not sed:
            if sed_filename:
                sed = self._read_sed('seds/ia/' + sed_filename)
            else:
                sed_filename = random.choice(self.ia_sed_files)
                sed = self._read_sed(sed_filename)
                
        # Trigger the lc generation function on this sed
        return self.gen_lc_from_sed(redshift, nites, sed, 'Ia', sed_filename, cosmo=cosmo)
    
    def gen_cc(self, redshift, nites, sed=None, sed_filename=None, cosmo=None):
        """
        Generate a SN-CC light curve
        
        :param redshfit: the redshift of the source
        :param nites: a list of night relative to peak you want to obtain a magnitude for
        :param sed: (optional) a dataframe containing the sed of the SN
        :param sed_filename: (optional) the filename of the sed of the SN
        :param cosmo: (optional) an astropy.cosmology instance
        :return: lc_dict: a dictionary with keys ['lc, 'obj_type', 'sed']
            - 'lc' contains a dataframe of the light from the object
            - 'obj_type' contains a string for the type of object. Will be 'II', 'Ibc', etc.
            - 'sed' contains the filename of the sed used
        """

        print(sed_filename)
        
        # If sed not specified, choose sed based on weight map
        if not sed:
            if sed_filename:
                sed = self._read_sed('seds/cc/' + sed_filename)
            else:
                sed_filename = random.choices(self.cc_sed_files, weights=self.cc_weights, k=1)[0]
                sed = self._read_sed(sed_filename)
        
        # Get the type of SN-CC
        obj_type = self.cc_info_df['SNTYPE'].values[self.cc_info_df['SED'].values == sed_filename.split('/')[-1].split('.')[0]][0]
        
        # Trigger the lc generation function on this sed
        return self.gen_lc_from_sed(redshift, nites, sed, obj_type, sed_filename, cosmo=cosmo)
                
    def gen_lc_from_sed(self, redshift, nites, sed, obj_type, sed_filename, cosmo=None):
        """
        Generate a light curve based on a time-series sed
        
        :param redshfit: the redshift of the source
        :param nites: a list of night relative to peak you want to obtain a magnitude for
        :param sed: a dataframe containing the sed of the object
        :param obj_type: a string for the type of object. Will be 'II', 'Ibc', Ia, etc.
        :param sed_filename: the filename of the sed of the object
        :param cosmo: (optional) an astropy.cosmology instance
        :return: lc_dict: a dictionary with keys ['lc, 'obj_type', 'sed']
            - 'lc' contains a dataframe of the light from the object
            - 'obj_type' contains a string for the type of object. Will be 'II', 'Ibc', etc.
            - 'sed' contains the filename of the sed used
        """
        
        # If nite not in the sed, set nite to the closest nite in the sed
        useable_nites = []
        sed_nites = np.unique(sed['NITE'].values)
        for nite in nites:
            if nite not in sed_nites:
                useable_nites.append(self._get_closest_nite(sed_nites, nite))
            else:
                useable_nites.append(nite)
        nites = useable_nites
            
        # Redshift the sed frequencies and wavelengths
        sed['WAVELENGTH_OBS'] = (1.0 + redshift) * sed['WAVELENGTH_REST'].values
        sed['FREQUENCY_OBS'] = sed['FREQUENCY_REST'].values * (1.0 + redshift)
        
        # Calculate distance modulus
        if not cosmo:
            cosmo = FlatLambdaCDM(H0=69.3 * u.km / (u.Mpc * u.s), 
                                  Om0=0.286, Tcmb0=2.725 * u.K, Neff=3.04, Ob0=0.0463)
        distance_modulus = self._get_distance_modulus(redshift, cosmo=cosmo)
        
        # Calculate k-correction at peak
        peak_sed = sed[sed['NITE'].values == self._get_closest_nite(np.unique(sed['NITE'].values), 0)].copy().reset_index(drop=True)
        k_corrections = [self._get_kcorrect(peak_sed, band, redshift) for band in self.bands]
        
        # On each nite, in each band, calculate the absolute mag
        output_data = []
        output_data_cols = ['NITE', 'BAND', 'MAG']
        for nite in nites:
            nite_sed = sed[sed['NITE'].values == nite].copy().reset_index(drop=True)
            
            # Apply factors to calculate absolute mag
            nite_sed['FLUX'] = (cosmo.luminosity_distance(redshift).value * 10 ** 6 / 10) ** 2 / (1 + redshift) * nite_sed['FLUX'].values
            nite_sed['FREQUENCY_REST'] = nite_sed['FREQUENCY_REST'].values / (1. + redshift)
            
            # Convert to AB Magnitude system
            norm_sed = nite_sed.copy()
            norm_sed['FLUX'] = 3631.0
            
            for band, k_correction in zip(self.bands, k_corrections):
                
                # Calculate the apparent magnitude
                norm = self._integrate_through_band(norm_sed, band, redshift, frame='REST')
                absolute_ab_mag = self._integrate_through_band(nite_sed, band, redshift, frame='REST') / norm
                output_data.append([nite, band, -2.5 * np.log10(absolute_ab_mag) + distance_modulus + k_correction])
                
        return {'lc': pd.DataFrame(data=output_data, columns=output_data_cols).replace(np.nan, 30.0, inplace=False),
                'obj_type': obj_type,
                'sed': sed_filename}

    

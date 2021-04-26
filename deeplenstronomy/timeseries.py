"""Generate light curves from time-series spectral energy distributions"""

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
        Initialize the LCGen object, download data if necessary, collect 
        necessary filter and sed information.
        
        Args:
            bands (str): comma-separated string of bands, i.e. 'g,r,i,z'
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

        # Load corrections
        self._load_corrections()
        
        # Interpolate the transmission curves
        self.norm_dict = {}
        for band in self.bands:
            transmission_frequency, transmission_wavelength = self.__read_passband(band)
            setattr(self, '{0}_transmission_frequency'.format(band), transmission_frequency)
            setattr(self, '{0}_transmission_wavelength'.format(band), transmission_wavelength)
            
            # Store normalizations
            lower_bound = eval('self._{0}_obs_frame_freq_min'.format(band))
            upper_bound = eval('self._{0}_obs_frame_freq_max'.format(band))        
            frequency_arr = np.linspace(upper_bound, lower_bound, 10000)
            norm_arr = np.ones(len(frequency_arr)) * 3631.0
            
            norm_sed = pd.DataFrame(data=np.vstack((norm_arr, frequency_arr)).T,
                                    columns=['FLUX', 'FREQUENCY_REST'])
            self.norm_dict[band] = self._integrate_through_band(norm_sed, band, 0.0, frame='REST')
        
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
        passband.loc[passband.shape[0]] = (1.e-9, 0.0)
        passband.loc[passband.shape[0]] = (4.e+9, 0.0)
        
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

        # Remove unrealistic wavelengths
        sed = sed[sed['WAVELENGTH_REST'].values > 10.0].copy().reset_index(drop=True)
        
        # Add new boundaries
        boundary_data = []
        for nite in np.unique(sed['NITE'].values):
            boundary_data.append((nite, 10.0, 0.0))
            boundary_data.append((nite, 25000.0, 0.0))
        sed = sed.append(pd.DataFrame(data=boundary_data, columns=['NITE', 'WAVELENGTH_REST', 'FLUX']))

        # Convert to frequency
        sed['FREQUENCY_REST'] = 2.99792458e18 / sed['WAVELENGTH_REST'].values

        # Normalize
        func = interp1d(sed['WAVELENGTH_REST'].values, sed['FLUX'].values)
        sed['FLUX'] = sed['FLUX'].values / quad(func, 10.0, 25000.0)[0]
        
        # Round nights to nearest int
        sed['NITE'] = sed['NITE'].values.round()
        
        return sed
    
    def _get_kcorrect(self, sed, band, redshift):
        """
        Calculate the K-Correction
        
        :param sed: the sed on the night of peak flux
        :param band: the single-letter band being used
        :param redshift: the redshift of the object
        :return: kcor: the k-correction to the absolute magnitude
        """
        kcorrect =  -2.5 * np.log10( 
                                   (self._integrate_through_band(sed, band, redshift, frame='OBS') /
                                    self._integrate_through_band(sed, band, redshift, frame='REST')) / (1.0 + redshift) )
        if np.isnan(kcorrect):
            # object is redshifted out of the passband
            return 99.0
        else:
            return kcorrect

    
    def _get_kcorrections(self, sed, sed_filename, redshift):
        """
        Cache the k-correction factors and return
        """
        attr_name = sed_filename.split('.')[0] + '-kcorrect_dict-' + str(redshift*100).split('.')[0]
        if hasattr(self, attr_name):
            return [getattr(self, attr_name)[b] for b in self.bands]
        else:
            peak_sed = sed[sed['NITE'].values == self._get_closest_nite(np.unique(sed['NITE'].values), 0)].copy().reset_index(drop=True)
            k_corrections = [self._get_kcorrect(peak_sed, band, redshift) for band in self.bands]
            setattr(self, attr_name, {b: k for b, k in zip(self.bands, k_corrections)})
            return k_corrections


    def _load_corrections(self):
        """
        10 degree polynomials to interpolate for magnitude calibration
        """
        coeff = {'g': np.array([4.51183210e+01, -4.44347636e+02, 1.83994410e+03, -4.13920560e+03,
                                5.43570479e+03, -4.11857064e+03, 1.58851514e+03, -1.24875350e+02,
                                -1.27356106e+02, 5.11527289e+01, -1.29985868e+00]), 
                 'r': np.array([17.15707984, -147.24752602, 498.8769753, -806.5297888,
                                493.38669116, 322.32930769, -768.87362622, 568.07310502,
                                -224.21701522, 57.16290538, -2.11245789]), 
                 'i': np.array([-1.52230194e+00, 2.13199036e+01, -1.32235571e+02, 4.71861410e+02,
                                -1.05833909e+03, 1.53795060e+03, -1.44717045e+03, 8.60925102e+02,
                                -3.12509126e+02, 7.12590111e+01, -2.99493586e+00]), 
                 'z': np.array([-2.24573756e+01, 2.39916362e+02, -1.09998790e+03, 2.82651442e+03,
                                -4.46372497e+03, 4.48084163e+03, -2.87704854e+03, 1.17947248e+03,
                                -3.16911098e+02, 6.50723854e+01, -3.28937692e+00])}
        
        self.corr = {b: np.poly1d(coeff_arr) for b, coeff_arr in coeff.items()}

        
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
        frequency_arr = sed['FREQUENCY_{0}'.format(frame)].values
        delta_frequencies = np.diff(frequency_arr) * -1.0
        integrand = eval("self.{0}_transmission_frequency(frequency_arr) * sed['FLUX'].values / frequency_arr".format(band))
        average_integrands = 0.5 * np.diff(integrand) + integrand[0:-1]
        res = np.sum(delta_frequencies * average_integrands)
        
        if np.isnan(res):
            # SED was redshifted out of passband
            return 1.e99
        else:
            return res

    
    def _get_closest_nite(self, unique_nites, nite):
        """
        Return the nite in the sed closest to a desired nite
        
        :param unique_nites: a set of the nights in an sed
        :param nite: the nite you wish to find the closest neighbor for
        :return: closest_nite: the closest nite in the sed to the given nite
        """
        
        ## If nite not in the sed, (but within range) set nite to the closest nite in the sed
        ## If nite is outside range, keep the same
        if nite > unique_nites.max() or nite < unique_nites.min():
            return nite
        else:
            return unique_nites[np.argmin(np.abs(nite - unique_nites))]

    def gen_variable(self, redshift, nite_dict, sed=None, sed_filename=None, cosmo=None):
        """
        Generate a random variable light curve

        Args:
            redshift (float): ignored 
            nite_dict (dict[str: List[int]]): (band, list of night relative to peak you want to obtain a magnitude for) pair for each band in survey    
            sed_filename (str): ignored
            cosmo (astropy.cosmology): ignored

        Returns:
            lc_dict: a dictionary with keys ['lc, 'obj_type', 'sed']
              - 'lc' contains a dataframe of the light from the object     
              - 'obj_type' contains a string for the type of object. Will always be 'Variable' here
              - 'sed' contains the filename of the sed used. Will always be 'Variable' here  
        """
        output_data_cols = ['NITE', 'BAND', 'MAG']
        output_data = []
        central_mag = random.uniform(12.0, 23.0)
        colors = {band: mag for band, mag in zip(self.bands, np.random.uniform(low=-2.0, high=2.0, size=len(self.bands)))}
        for band in self.bands:
            for nite in nite_dict[band]:
                central_mag = random.uniform(central_mag - 1.0, central_mag + 1.0)
                output_data.append([nite, band, central_mag + colors[band]])

        return {'lc': pd.DataFrame(data=output_data, columns=output_data_cols),
                'obj_type': 'Variable',
                'sed': 'Variable'}
    
    def gen_flat(self, redshift, nite_dict, sed=None, sed_filename=None, cosmo=None):
        """
        Generate a random flat light curve.
        
        Args:
            redshift (float): ignored
            nite_dict (dict[str: List[int]]): (band, list of night relative to peak you want to obtain a magnitude for) pair for each band in survey
            sed_filename (str): ignored
            cosmo (astropy.cosmology): ignored

        Returns:
            lc_dict: a dictionary with keys ['lc, 'obj_type', 'sed']
              - 'lc' contains a dataframe of the light from the object
              - 'obj_type' contains a string for the type of object. Will always be 'Flat' here
              - 'sed' contains the filename of the sed used. Will always be 'Flat' here      
        """
        output_data_cols = ['NITE', 'BAND', 'MAG']
        central_mag = random.uniform(12.0, 23.0)
        mags = {band: mag for band, mag in zip(self.bands, central_mag + np.random.uniform(low=-2.0, high=2.0, size=len(self.bands)))}
        output_data = []
        for band in self.bands:
            for nite in nite_dict[band]:
                output_data.append([nite, band, mags[band]])

        return {'lc': pd.DataFrame(data=output_data, columns=output_data_cols),
                'obj_type': 'Flat',
                'sed': 'Flat'}

    def gen_static(self, redshift, nite_dict, sed=None, sed_filename=None, cosmo=None):
        """
        Make a static source capable of having time-series data by introducing a mag=99 source
        on each NITE of the simulation.

        Args:
            redshift (float): ignored 
            nite_dict (dict[str: List[int]]): (band, list of night relative to peak you want to obtain a magnitude for) pair for each band in survey
            sed_filename (str): ignored                                                                                                                                                               
            cosmo (astropy.cosmology): ignored                                                                                                                                                                                            
        Returns:
            lc_dict: a dictionary with keys ['lc, 'obj_type', 'sed']
              - 'lc' contains a dataframe of the light from the object
              - 'obj_type' contains a string for the type of object. Will always be 'Static' here
              - 'sed' contains the filename of the sed used. Will always be 'Flat' here     
        """
        output_data_cols = ['NITE', 'BAND', 'MAG']
        central_mag = 99.0
        mags = {band: central_mag for band in self.bands}
        output_data = []
        for band in self.bands:
            for nite in nite_dict[band]:
                output_data.append([nite, band, mags[band]])

        return {'lc': pd.DataFrame(data=output_data, columns=output_data_cols),
                'obj_type': 'Static',
                'sed': 'Static'}


        
    def gen_variablenoise(self, redshift, nite_dict, sed=None, sed_filename=None, cosmo=None):
        """ 
        Generate a variable light curve with small random noise

        Args:
            redshift (float): ignored
            nite_dict (dict[str: List[int]]): (band, list of night relative to peak you want to obtain a magnitude for) pair for each band in survey
            sed_filename (str): ignored
            cosmo (astropy.cosmology): ignored

        Returns:
            lc_dict: a dictionary with keys ['lc, 'obj_type', 'sed']
              - 'lc' contains a dataframe of the light from the object
              - 'obj_type' contains a string for the type of object. Will always be 'VariableNoise' here
              - 'sed' contains the filename of the sed used. Will always be 'VariableNoise' here              
        """
        noiseless_lc_dict = self.gen_variable(redshift, nite_dict)
        noise = np.random.normal(loc=0, scale=0.25, size=noiseless_lc_dict['lc'].shape[0])
        noiseless_lc_dict['lc']['MAG'] = noiseless_lc_dict['lc']['MAG'].values + noise
        noiseless_lc_dict['obj_type'] = 'VariableNoise'
        noiseless_lc_dict['sed'] = 'VariableNoise'
        return noiseless_lc_dict

    
    def gen_flatnoise(self, redshift, nite_dict, sed=None, sed_filename=None, cosmo=None):
        """
        Generate a flat light curve will small random noise

        Args:
            redshift (float): ignored
            nite_dict (dict[str: List[int]]): (band, list of night relative to peak you want to obtain a magnitude for) pair for each band in survey
            sed_filename (str): ignored
            cosmo (astropy.cosmology): ignored

        Returns:
            lc_dict: a dictionary with keys ['lc, 'obj_type', 'sed']
              - 'lc' contains a dataframe of the light from the object
              - 'obj_type' contains a string for the type of object. Will always be 'FlatNoise' here
              - 'sed' contains the filename of the sed used. Will always be 'FlatNoise' here              
        """
        noiseless_lc_dict = self.gen_flat(redshift, nite_dict)
        noise = np.random.normal(loc=0, scale=0.25, size=noiseless_lc_dict['lc'].shape[0])
        noiseless_lc_dict['lc']['MAG'] = noiseless_lc_dict['lc']['MAG'].values + noise
        noiseless_lc_dict['obj_type'] = 'FlatNoise'
        noiseless_lc_dict['sed'] = 'FlatNoise'
        return noiseless_lc_dict
        
    def gen_user(self, redshift, nite_dict, sed=None, sed_filename=None, cosmo=None):
        """
        Generate a light curve from a user-specidied SED

        Args:
            redshift (float): the redshift of the source
            nite_dict (dict[str: List[int]]): (band, list of night relative to peak you want to obtain a magnitude for) pair for each band in survey 
            sed (None or pandas.DataFrame, optional, default=None): a dataframe containing the sed of the SN 
            sed_filename (str): filename containing the time-series sed you want to use 
            cosmo (astropy.cosmology): an astropy.cosmology instance used for distance calculations

        Returns:
            lc_dict: a dictionary with keys ['lc, 'obj_type', 'sed']
              - 'lc' contains a dataframe of the light from the object
              - 'obj_type' contains a string for the type of object. Will always be <sed_filename> here 
              - 'sed' contains the filename of the sed used  
        """
        if sed is None:
            if sed_filename.startswith('seds/user/'):
                attr_name = sed_filename.split('.')[0]
            else:
                attr_name = 'seds/user/' + sed_filename.split('.')[0]
                sed_filename = 'seds/user/' + sed_filename
                
            if hasattr(self, attr_name):
                sed = getattr(self, attr_name)
            else:
                sed = self._read_sed(sed_filename)
                setattr(self, attr_name, sed)

        return self.gen_lc_from_sed(redshift, nite_dict, sed, sed_filename, sed_filename, cosmo=cosmo)

    def gen_kn(self, redshift, nite_dict, sed=None, sed_filename=None, cosmo=None):
        """
        Generate a GW170817-like light curve.

        Args:
            redshift (float): the redshift of the source
            nite_dict (dict[str: List[int]]): (band, list of night relative to peak you want to obtain a magnitude for) pair for each band in survey 
            sed (None or pandas.DataFrame, optional, default=None): a dataframe containing the sed of the SN 
            sed_filename (str): filename containing the time-series sed you want to use 
            cosmo (astropy.cosmology): an astropy.cosmology instance used for distance calculations

        Returns:
            lc_dict: a dictionary with keys ['lc, 'obj_type', 'sed']
              - 'lc' contains a dataframe of the light from the object
              - 'obj_type' contains a string for the type of object. Will always be KN here 
              - 'sed' contains the filename of the sed used  
        """

        sed_filename = 'seds/kn/kn.SED'
        if sed is None:
            attr_name = sed_filename.split('.')[0]
            if hasattr(self, attr_name):
                sed = getattr(self, attr_name)
            else:
                sed = self._read_sed(sed_filename)
                setattr(self, attr_name, sed)
                
        return self.gen_lc_from_sed(redshift, nite_dict, sed, 'KN', sed_filename, cosmo=cosmo)
    
    def gen_ia(self, redshift, nite_dict, sed=None, sed_filename=None, cosmo=None):
        """
        Generate a SN-Ia light curve.

        Args:
            redshift (float): the redshift of the source
            nite_dict (dict[str: List[int]]): (band, list of night relative to peak you want to obtain a magnitude for) pair for each band in survey 
            sed (None or pandas.DataFrame, optional, default=None): a dataframe containing the sed of the SN 
            sed_filename (str): filename containing the time-series sed you want to use 
            cosmo (astropy.cosmology): an astropy.cosmology instance used for distance calculations

        Returns:
            lc_dict: a dictionary with keys ['lc, 'obj_type', 'sed']
              - 'lc' contains a dataframe of the light from the object
              - 'obj_type' contains a string for the type of object. Will always be Ia here 
              - 'sed' contains the filename of the sed used  
        """
        
        # Read rest-frame sed if not supplied as argument
        if sed is None:
            if sed_filename is None:
                sed_filename = random.choice(self.ia_sed_files)
            
            if sed_filename.startswith('seds/ia/'):
                attr_name = sed_filename.split('.')[0]
            else:
                attr_name = 'seds/ia/' + sed_filename.split('.')[0]
                sed_filename = 'seds/ia/' + sed_filename

            if hasattr(self, attr_name):
                sed = getattr(self, attr_name)
            else:
                sed = self._read_sed(sed_filename)
                setattr(self, attr_name, sed)
                
        # Trigger the lc generation function on this sed
        return self.gen_lc_from_sed(redshift, nite_dict, sed, 'Ia', sed_filename, cosmo=cosmo)
    
    def gen_cc(self, redshift, nite_dict, sed=None, sed_filename=None, cosmo=None):
        """
        Generate a SN-CC light curve
        
        Args:
            redshift (float): the redshift of the source
            nite_dict (dict[str: List[int]]): (band, list of night relative to peak you want to obtain a magnitude for) pair for each band in survey 
            sed (None or pandas.DataFrame, optional, default=None): a dataframe containing the sed of the SN 
            sed_filename (str): filename containing the time-series sed you want to use 
            cosmo (astropy.cosmology): an astropy.cosmology instance used for distance calculations

        Returns:
            lc_dict: a dictionary with keys ['lc, 'obj_type', 'sed']
              - 'lc' contains a dataframe of the light from the object
              - 'obj_type' contains a string for the type of object. Will be 'II', 'Ibc, etc. 
              - 'sed' contains the filename of the sed used  
        """

        # If sed not specified, choose sed based on weight map
        if sed is None:
            if sed_filename is None:
                sed_filename = random.choices(self.cc_sed_files, weights=self.cc_weights, k=1)[0]

            if sed_filename.startswith('seds/cc/'):
                attr_name = sed_filename.split('.')[0]
            else:
                attr_name = 'seds/cc/' + sed_filename.split('.')[0]
                sed_filename = 'seds/cc/' + sed_filename
                    
            if hasattr(self, attr_name):
                sed = getattr(self, attr_name)
            else:
                sed = self._read_sed(sed_filename)
                setattr(self, attr_name, sed)
        
        # Get the type of SN-CC
        obj_type = self.cc_info_df['SNTYPE'].values[self.cc_info_df['SED'].values == sed_filename.split('/')[-1].split('.')[0]][0]
        
        # Trigger the lc generation function on this sed
        return self.gen_lc_from_sed(redshift, nite_dict, sed, obj_type, sed_filename, cosmo=cosmo)
                
    def gen_lc_from_sed(self, redshift, nite_dict, sed, obj_type, sed_filename, cosmo=None):
        """
        Generate a light curve based on a time-series sed.
        
        Args:
            redshift (float): the redshift of the source
            nite_dict (dict[str: List[int]]): (band, list of night relative to peak you want to obtain a magnitude for) pair for each band in survey 
            sed (None or pandas.DataFrame, optional, default=None): a dataframe containing the sed of the SN 
            sed_filename (str): filename containing the time-series sed you want to use 
            cosmo (astropy.cosmology): an astropy.cosmology instance used for distance calculations

        Returns:
            lc_dict: a dictionary with keys ['lc, 'obj_type', 'sed']
              - 'lc' contains a dataframe of the light from the object
              - 'obj_type' contains a string for the type of object. 
              - 'sed' contains the filename of the sed used  
        """
        
        # Adjust nites
        nites = {}
        sed_nites = np.unique(sed['NITE'].values)
        for band, cad_nites_ in nite_dict.items():

            # evaluate on a wider grid than the cadence
            cad_nites = np.linspace(min(cad_nites_) - 50, max(cad_nites_) + 50, 5 * len(cad_nites_)).round().astype(int)
            _, cad_idx = np.unique(cad_nites, return_index=True)
            cad_nites = cad_nites[np.sort(cad_idx)]
            
            useable_nites = []
            for nite in cad_nites:
                if nite not in sed_nites:
                    useable_nites.append(self._get_closest_nite(sed_nites, nite))
                else:
                    useable_nites.append(nite)
            nites[band] = useable_nites
            
        # Redshift the sed frequencies and wavelengths
        sed['WAVELENGTH_OBS'] = (1.0 + redshift) * sed['WAVELENGTH_REST'].values
        sed['FREQUENCY_OBS'] = sed['FREQUENCY_REST'].values / (1.0 + redshift)
        
        # Calculate distance modulus
        if not cosmo:
            cosmo = FlatLambdaCDM(H0=69.3 * u.km / (u.Mpc * u.s), 
                                  Om0=0.286, Tcmb0=2.725 * u.K, Neff=3.04, Ob0=0.0463)
        distance_modulus = self._get_distance_modulus(redshift, cosmo=cosmo)
        
        # Calculate k-correction at peak
        k_corrections = self._get_kcorrections(sed, sed_filename, redshift)
        
        # On each nite, in each band, calculate the absolute mag
        output_data = []
        output_data_cols = ['NITE', 'BAND', 'MAG']
        
        for band, k_correction in zip(self.bands, k_corrections):
            
            for nite in nites[band]:
                nite_sed = sed[sed['NITE'].values == nite].copy().reset_index(drop=True)
                
                # Flux is zero if requested nite is noe in sed
                if len(nite_sed) == 0:
                    output_data.append([nite, band, 99.0])
                    continue
                
                # Apply factors to calculate absolute mag
                nite_sed['FLUX'] = (cosmo.luminosity_distance(redshift).value * 10 ** 6 / 10) ** 2 / (1 + redshift) * nite_sed['FLUX'].values
                nite_sed['FREQUENCY_REST'] = nite_sed['FREQUENCY_REST'].values / (1. + redshift)
            
                # Calculate the apparent magnitude
                absolute_ab_mag = self._integrate_through_band(nite_sed, band, redshift, frame='REST') / self.norm_dict[band]
                output_data.append([nite, band, -2.5 * np.log10(absolute_ab_mag) + distance_modulus + k_correction + self.corr[band](redshift)])
                
                # Output
                #print("Nite:", nite, "\tBand:", band, "\tBase: %.2f" %(-2.5 * np.log10(absolute_ab_mag)), "\tDist:", round(distance_modulus, 2), "\tKC:", round(k_correction, 2), '\tMAG:', round(-2.5 * np.log10(absolute_ab_mag) + distance_modulus + k_correction, 2))
                
        return {'lc': pd.DataFrame(data=output_data, columns=output_data_cols).replace(np.inf, 99.0, inplace=False).replace(np.nan, 99.0, inplace=False),
                'obj_type': obj_type,
                'sed': sed_filename}

# functions for performing calculations

import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u


def convert_apmag_to_abmag(mag, d):
    #d must be in parsecs
    #mag must be greater than zero
    M = mag - 5 * np.log10(d / 10)
    return -1.0 * M

def convert_z_to_d(z, H0=70, Tcmb0=2.725, Om0=0.3):
    #also works on arrays of z values
    cosmo = FlatLambdaCDM(H0=H0 * u.km / u.s / u.Mpc, Tcmb0=Tcmb0 * u.K, Om0=Om0)
    luminosity_distance = cosmo.luminosity_distance(z)
    #returns distance in parsecs
    return luminosity_distance.value * 10 ** 6

def convert_mjd_to_nite(mjd):
    return int(mjd + 0.5)

import numpy as np
import os

file_dir = os.path.join( os.path.dirname( __file__ ), '../../2dpdfs' )

class StochasticNoise(object):
  "Returns an array of specified size of randomly selected values of stochastic seeing and sky-brightness from a list."
  def __init__(self, size, pdf):
    rand_idx = np.random.randint(len(pdf),size=size)
    self.seeing = pdf[rand_idx,0]
    self.sky_brightness = pdf[rand_idx,1]

def noise_des(band,directory=file_dir):
    """
    Simulated noise profile for a given DES band.
    Output: dict containing survey characteristics:
    DES_survey_noise = {'read_noise': float
                        'pixel_scale': float
                        'ccd_gain': float
                        'exposure_time': float
                        'magnitude_zero_point': float
                        'num_exposures': int
                        'psf_type': string
                        'seeing': float
                        'sky_brightness': float}
    """
    if band == 'g':
        pdf = np.loadtxt("%s/2dg_DES.txt" %directory)
        rand_idx = np.random.randint(len(pdf))
        seeing = pdf[rand_idx,0]
        sky_brightness = pdf[rand_idx,1]
        DES_survey_noise = {'read_noise': 10.,
                            'pixel_scale': 0.263,
                            'ccd_gain': 4.5,
                            'exposure_time': 90.,
                            'magnitude_zero_point': 30.,
                            'num_exposures': 10,
                            'psf_type': 'GAUSSIAN',
                            'seeing': seeing,
                            'sky_brightness': sky_brightness}

    elif band == 'r':
        pdf = np.loadtxt("%s/2dr_DES.txt" %directory)
        rand_idx = np.random.randint(len(pdf))
        seeing = pdf[rand_idx,0]
        sky_brightness = pdf[rand_idx,1]
        DES_survey_noise = {'read_noise': 10.,
                            'pixel_scale': 0.263,
                            'ccd_gain': 4.5,
                            'exposure_time': 90.,
                            'magnitude_zero_point': 30.,
                            'num_exposures': 10,
                            'psf_type': 'GAUSSIAN',
                            'seeing': seeing,
                            'sky_brightness': sky_brightness}

    elif band == 'i':
        pdf = np.loadtxt("%s/2di_DES.txt" %directory)
        rand_idx = np.random.randint(len(pdf))
        seeing = pdf[rand_idx,0]
        sky_brightness = pdf[rand_idx,1]
        DES_survey_noise = {'read_noise': 10.,
                            'pixel_scale': 0.263,
                            'ccd_gain': 4.5,
                            'exposure_time': 90.,
                            'magnitude_zero_point': 30.,
                            'num_exposures': 10,
                            'psf_type': 'GAUSSIAN',
                            'seeing': seeing,
                            'sky_brightness': sky_brightness}
    else:
        raise ValueError('%s band is not supported.' % band)

    return DES_survey_noise

def noise_lsst(band,directory=file_dir):
    # Generates nobs simulated noise profiles.
    """
    Simulated noise profile for a given LSST band.
    Output: dict containing survey characteristics:
    LSST_survey_noise = {'read_noise': float
                        'pixel_scale': float
                        'ccd_gain': float
                        'exposure_time': float
                        'magnitude_zero_point': float
                        'num_exposures': int
                        'psf_type': string
                        'seeing': float
                        'sky_brightness': float}
    """
    if band == 'g':
        pdf = np.loadtxt("%s/2dg_LSST.txt" %directory)
        rand_idx = np.random.randint(len(pdf))
        seeing = pdf[rand_idx,0]
        sky_brightness = pdf[rand_idx,1]
        LSST_survey_noise = {'read_noise': 10.,
                            'pixel_scale': 0.2,
                            'ccd_gain': 4.5,
                            'exposure_time': 30.,
                            'magnitude_zero_point': 30.,
                            'num_exposures': 100,
                            'psf_type': 'GAUSSIAN',
                            'seeing': seeing,
                            'sky_brightness': sky_brightness}

    elif band == 'r':
        pdf = np.loadtxt("%s/2dr_LSST.txt" %directory)
        rand_idx = np.random.randint(len(pdf))
        seeing = pdf[rand_idx,0]
        sky_brightness = pdf[rand_idx,1]
        LSST_survey_noise = {'read_noise': 10.,
                            'pixel_scale': 0.2,
                            'ccd_gain': 4.5,
                            'exposure_time': 30.,
                            'magnitude_zero_point': 30.,
                            'num_exposures': 200,
                            'psf_type': 'GAUSSIAN',
                            'seeing': seeing,
                            'sky_brightness': sky_brightness}

    elif band == 'i':
        pdf = np.loadtxt("%s/2di_LSST.txt" %directory)
        rand_idx = np.random.randint(len(pdf))
        seeing = pdf[rand_idx,0]
        sky_brightness = pdf[rand_idx,1]
        LSST_survey_noise = {'read_noise': 10.,
                            'pixel_scale': 0.2,
                            'ccd_gain': 4.5,
                            'exposure_time': 30.,
                            'magnitude_zero_point': 30.,
                            'num_exposures': 200,
                            'psf_type': 'GAUSSIAN',
                            'seeing': seeing,
                            'sky_brightness': sky_brightness}

    else:
        raise ValueError('%s band is not supported.' % band)
    return LSST_survey_noise


def noise_cfht(band,directory=file_dir):
    # Generates nobs simulated noise profiles.
    """
    Simulated noise profile for a given CFHT band.
    Output: dict containing survey characteristics:
    CFHT_survey_noise = {'read_noise': float
                        'pixel_scale': float
                        'ccd_gain': float
                        'exposure_time': float
                        'magnitude_zero_point': float
                        'num_exposures': int
                        'psf_type': string
                        'seeing': float
                        'sky_brightness': float}
    """
    if band =='g':
        pdf = np.loadtxt("%s/2dg_CFHT.txt" %directory)
        rand_idx = np.random.randint(len(pdf))
        seeing = pdf[rand_idx,0]
        sky_brightness = pdf[rand_idx,1]
        CFHT_survey_noise = {'read_noise': 10.,
                            'pixel_scale': 0.187,
                            'ccd_gain': 8.6,
                            'exposure_time': 3500.,
                            'magnitude_zero_point': 30.,
                            'num_exposures': 1,
                            'psf_type': 'GAUSSIAN',
                            'seeing': seeing,
                            'sky_brightness': sky_brightness}


    elif band == 'r':
        pdf = np.loadtxt("%s/2dr_CFHT.txt" %directory)
        rand_idx = np.random.randint(len(pdf))
        seeing = pdf[rand_idx,0]
        sky_brightness = pdf[rand_idx,1]
        CFHT_survey_noise = {'read_noise': 10.,
                            'pixel_scale': 0.187,
                            'ccd_gain': 8.6,
                            'exposure_time': 5500.,
                            'magnitude_zero_point': 30.,
                            'num_exposures': 1,
                            'psf_type': 'GAUSSIAN',
                            'seeing': seeing,
                            'sky_brightness': sky_brightness}

    elif band == 'i':
        pdf = np.loadtxt("%s/2di_CFHT.txt" %directory)
        rand_idx = np.random.randint(len(pdf))
        seeing = pdf[rand_idx,0]
        sky_brightness = pdf[rand_idx,1]
        CFHT_survey_noise = {'read_noise': 10.,
                            'pixel_scale': 0.187,
                            'ccd_gain': 8.6,
                            'exposure_time': 5500.,
                            'magnitude_zero_point': 30.,
                            'num_exposures': 1,
                            'psf_type': 'GAUSSIAN',
                            'seeing': seeing,
                            'sky_brightness': sky_brightness}

    else:
        raise ValueError('%s band is not supported.' % band)

    return CFHT_survey_noise


def survey_noise(survey_name, band, directory=file_dir):
    "Specify survey name and band"
    if survey_name == 'DES':
         survey_noise = noise_des(band,directory)
    elif survey_name == 'LSST':
         survey_noise = noise_lsst(band,directory)
    elif survey_name == 'CFHT':
         survey_noise = noise_cfht(band,directory)
    else:
         raise ValueError('%s is not supported.' % survey_name)
    return survey_noise

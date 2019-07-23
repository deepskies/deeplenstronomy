import numpy as np
import pickle
import lenstronomy.Util.data_util as data_util
import lenstronomy.Util.util as util
import lenstronomy.Plots.plot_util as plot_util


class StochasticNoise(object):
  "Returns an array of specified size of randomly selected values of stochastic seeing and sky-brightness from a list."
  def __init__(self, size, pdf):
    rand_idx = np.random.randint(len(pdf),size=size)
    self.seeing = pdf[rand_idx,0]
    self.sky_brightness = pdf[rand_idx,1]

def DES_noise(nobs,band=None,directory='2dpdfs'):
    # Generates nobs simulated noise profiles.
    DES_camera = {'read_noise': 10,
                    'pixel_scale': 0.263,
                    'ccd_gain': 4.5
                    }

    if band == 'g':
        # Static noise quantities
        DES_band_obs = {'exposure_time': 90.,
                           'magnitude_zero_point': 30,
                           'num_exposures': 10,
                           'psf_type': 'GAUSSIAN'}
        twodg = pickle.load(open("%s/2dg_DES.pkl" %directory,'rb'))
        DES_stochastic_noise = StochasticNoise(nobs,twodg)

    elif band == 'r':
        DES_band_obs = {'exposure_time': 90.,
                           'magnitude_zero_point': 30,
                           'num_exposures': 10,
                           'psf_type': 'GAUSSIAN'}
        twodr = pickle.load(open("%s/2dr_DES.pkl" %directory,'rb'))
        DES_stochastic_noise = StochasticNoise(nobs,twodr)

    elif band == 'i':
        DES_band_obs = {'exposure_time': 90.,
                           'magnitude_zero_point': 30,
                           'num_exposures': 10,
                           'psf_type': 'GAUSSIAN'}
        twodi = pickle.load(open("%s/2di_DES.pkl" %directory,'rb'))
        DES_stochastic_noise = StochasticNoise(nobs,twodi)
    else:
        raise ValueError('band secified as %s is not supported.' % band)

    # twodg[:,0]) seeing
    # twodg[:,1]) sky brightness

    # Use these if you are using Python 3 or have issues with pickle reading the file
    # twodg = pickle.load(open("%s/2dg_LSST.pkl" %directory,'rb'),encoding='latin1')
    # twodr = pickle.load(open("%s/2dr_LSST.pkl" %directory,'rb'),encoding='latin1')
    # twodi = pickle.load(open("%s/2di_LSST.pkl" %directory,'rb'),encoding='latin1')

    DES_static_noise = util.merge_dicts(DES_camera, DES_band_obs)
    DES_noise_array = []
    for i in range(nobs):
        DES_stochastic_noise_dict = {'seeing': DES_stochastic_noise.seeing[i],
                              'sky_brightness': DES_stochastic_noise.sky_brightness[i]}
        DES_noise_array.append(util.merge_dicts(DES_static_noise, DES_stochastic_noise_dict))
    return DES_noise_array

def LSST_noise(nobs,band=None,directory='2dpdfs'):
    # Generates nobs simulated noise profiles.
    LSST_camera = {'read_noise': 10,
                    'pixel_scale': 0.2,
                    'ccd_gain': 4.5
                    }

    if band == 'g':
        # Static noise quantities
        LSST_band_obs = {'exposure_time': 30.,
                           'magnitude_zero_point': 30,
                           'num_exposures': 100,
                           'psf_type': 'GAUSSIAN',
                          }
        twodg = pickle.load(open("%s/2dg_LSST.pkl" %directory,'rb'))
        LSST_stochastic_noise = StochasticNoise(nobs,twodg)

    elif band == 'r':
        LSST_band_obs = {'exposure_time': 60.,
                           'magnitude_zero_point': 30,
                           'num_exposures': 100,
                           'psf_type': 'GAUSSIAN'}
        twodr = pickle.load(open("%s/2dr_LSST.pkl" %directory,'rb'))
        LSST_stochastic_noise = StochasticNoise(nobs,twodr)

    elif band == 'i':
        LSST_band_obs = {'exposure_time': 60.,
                           'magnitude_zero_point': 30,
                           'num_exposures': 100,
                           'psf_type': 'GAUSSIAN'}

        directory = "2dpdfs"
        twodi = pickle.load(open("%s/2di_LSST.pkl" %directory,'rb'))
        LSST_stochastic_noise = StochasticNoise(nobs,twodi)

    else:
        print('specify band!')
        exit()

    # twodg[:,0]) seeing
    # twodg[:,1]) sky brightness

    # Use these if you are using Python 3 or have issues with pickle reading the file
    # twodg = pickle.load(open("%s/2dg_LSST.pkl" %directory,'rb'),encoding='latin1')
    # twodr = pickle.load(open("%s/2dr_LSST.pkl" %directory,'rb'),encoding='latin1')
    # twodi = pickle.load(open("%s/2di_LSST.pkl" %directory,'rb'),encoding='latin1')

    LSST_static_noise = util.merge_dicts(LSST_camera, LSST_band_obs)
    LSST_noise_array = []
    for i in range(nobs):
        LSST_stochastic_noise_dict = {'seeing': LSST_stochastic_noise.seeing[i],
                              'sky_brightness': LSST_stochastic_noise.sky_brightness[i]}
        LSST_noise_array.append(util.merge_dicts(LSST_static_noise, LSST_stochastic_noise_dict))
    return LSST_noise_array

def CFHT_noise(nobs=1,band=None,directory='2dpdfs'):
    # Generates nobs simulated noise profiles.
    CFHT_camera = {'read_noise': 10,
                    'pixel_scale': 0.187,
                    'ccd_gain': 8.6
                    }
    if band == 'g':
        # Static noise quantities
        CFHT_band_obs = {'exposure_time': 3500.,
                           'magnitude_zero_point': 30,
                           'num_exposures': 1,
                           'psf_type': 'GAUSSIAN'}
        twodg = pickle.load(open("%s/2dg_CFHT.pkl" %directory,'rb'))
        CFHT_stochastic_noise = StochasticNoise(nobs,twodg)

    elif band == 'r':
        CFHT_band_obs = {'exposure_time': 5500.,
                           'magnitude_zero_point': 30,
                           'num_exposures': 1,
                           'psf_type': 'GAUSSIAN'}
        twodr = pickle.load(open("%s/2dr_CFHT.pkl" %directory,'rb'))
        CFHT_stochastic_noise = StochasticNoise(nobs,twodr)

    elif band == 'i':
        CFHT_band_obs = {'exposure_time': 5500.,
                           'magnitude_zero_point': 30,
                           'num_exposures': 1,
                           'psf_type': 'GAUSSIAN'}
        twodi = pickle.load(open("%s/2di_CFHT.pkl" %directory,'rb'))
        CFHT_stochastic_noise = StochasticNoise(nobs,twodi)

    else:
        print('specify band!')
        exit()

    # twodg[:,0]) seeing
    # twodg[:,1]) sky brightness

    # Use these if you are using Python 3 or have issues with pickle reading the file
    # twodg = pickle.load(open("%s/2dg_LSST.pkl" %directory,'rb'),encoding='latin1')
    # twodr = pickle.load(open("%s/2dr_LSST.pkl" %directory,'rb'),encoding='latin1')
    # twodi = pickle.load(open("%s/2di_LSST.pkl" %directory,'rb'),encoding='latin1')

    CFHT_static_noise = util.merge_dicts(CFHT_camera, CFHT_band_obs)
    CFHT_noise_array = []
    for i in range(nobs):
        CFHT_stochastic_noise_dict = {'seeing': CFHT_stochastic_noise.seeing[i],
                              'sky_brightness': CFHT_stochastic_noise.sky_brightness[i]}
        CFHT_noise_array.append(util.merge_dicts(CFHT_static_noise, CFHT_stochastic_noise_dict))

    return CFHT_noise_array

def SurveyNoise(Name, band, nobs, directory='2dpdfs'):
    if Name == 'DES':
         Noise = DES_noise(nobs,band,directory)
    elif Name == 'LSST':
         Noise = LSST_noise(nobs,band,directory)
    elif Name == 'CFHT':
         Noise = CFHT_noise(nobs,band,directory)
    return Noise

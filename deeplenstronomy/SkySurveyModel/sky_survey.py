import numpy as np
import os
import yaml

pdfs_dir = os.path.join(os.path.dirname(__file__), '../../2dpdfs')
characteristics_dir = os.path.join(os.path.dirname(__file__),
                                   '../../survey_characteristics')


class StochasticNoise(object):
    """
    Returns an array of specified size of randomly selected values of stochastic
    seeing and sky-brightness from a list.
    """
    def __init__(self, size, pdf):
        rand_idx = np.random.randint(len(pdf), size=size)
        self.seeing = pdf[rand_idx, 0]
        self.sky_brightness = pdf[rand_idx, 1]


def noise_from_yaml(survey, band, pdfs_dir=pdfs_dir,
                    yaml_dir=characteristics_dir):
    # Generates nobs simulated noise profiles.
    """
    Loads noise configuration from yaml file and 2d pdf file
    """
    try:
        pdf = np.loadtxt("%s/2d%s_%s.txt" % (pdfs_dir, band, survey))
        rand_idx = np.random.randint(len(pdf))
        seeing = pdf[rand_idx, 0]
        sky_brightness = pdf[rand_idx, 1]
        yaml_file = '%s/%s_%s.yaml' % (yaml_dir, band, survey)
        with open(yaml_file, 'r') as config_file:
            survey_noise = yaml.safe_load(config_file)
        survey_noise['seeing'] = seeing
        survey_noise['sky_brightness'] = sky_brightness
    except FileNotFoundError:
        raise ValueError('%s band in survey %s is not supported.' %
                         (band, survey))

    return survey_noise


def survey_noise(survey_name, band, directory=pdfs_dir):
    """Specify survey name and band"""
    survey_noise = noise_from_yaml(survey_name, band, directory)
    return survey_noise

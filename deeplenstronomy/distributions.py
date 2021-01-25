"""Contains all available distributions. Utilize the functions in this module
by using the `DISTRIBUTION` keyword in your configuration file. As an example:

```
seeing:
    DISTRIBUTION:
        NAME: uniform # function name to call
        PARAMETERS:
            minimum: 0 # value to set for function argument
            maximim: 6 # value to set for function argument
```

"""

import numpy as np
import random
from scipy.stats import poisson

## Single parameter sampling distributions

def uniform(minimum, maximum, bands=''):
    """
    Return a samle from a uniform probability distribution
    on the interval [`minimum`, `maximum`]

    Args:
        minimum (float or int): The minimum of the interval to sample
        maximum (float or int): The maximum of the interval to sample

    Returns:
        A sample of the specified uniform distribution for each band in the simulation
    """
    draw = random.uniform(minimum, maximum)
    return [draw] * len(bands.split(','))

def uniform_int(minimum, maximum, bands=''):
    """
    Return a samle from a uniform probability distribution
    on the interval [`minimum`, `maximum`] rounded to the nearest integer

    Args:
        minimum (int): The minimum of the interval to sample
        maximum (int): The maximum of the interval to sample

    Returns:
        A rounded sample of the specified uniform distribution for each band in the simulation
    """
    draw = round(random.uniform(minimum, maximum))
    return [draw] * len(bands.split(','))

def normal(mean, std, bands=''):
    """
    Return a samle from a normal probability distribution
    with specifeid mean and standard deviation

    Args:
        mean (float or int): The mean of the normal distribution to sample
        std (float or int): The standard deviation of the normal distribution to sample

    Returns:
        A sample of the specified normal distribution for each band in the simulation
    """
    draw = np.random.normal(loc=mean, scale=std)
    return [draw] * len(bands.split(','))

def lognormal(mean, sigma, bands=''):
    """
    Return a samle from a lognormal probability distribution
    with specifeid mean and standard deviation

    Args:
        mean (float or int): The mean of the lognormal distribution to sample
        sigma (float or int): The standard deviation of the lognormal distribution to sample 

    Returns:
        A sample of the specified lognormal distribution for each band in the simulation
    """
    draw = np.random.lognormal(mean=mean, sigma=sigma)
    return [draw] * len(bands.split(','))

def delta_function(value, bands=''):
    """
    Use a delta function to set a specific value. Alternatively you can directly set the
    value of a parameter if it is going to be constant. This functionality is useful if
    you want to avoid deleting the `DISTRIBUTION` entry for a parameter in your
    configuration file.

    Args:
        value : The value to set for this parameter

    Returns:
        A list of values with one value for each band in the simulation
    """
    return [value] * len(bands.split(','))

def symmetric_uniform_annulus(r1, r2, bands=''):
    """
    Return a sample from a uniform probability distribtuion on the interval
    [`-r2`, `-r1`] U [`r1`, `r2`]. Useful for setting `center_x`, `center_y`, `sep`, etc.  while
    avoiding zero values in the sample.
    
    Args:
        r1 (float or int): The minimum radius of the symmetric annulus
        r2 (float or int): The maximum radius of the symmetric annulus

    Returns:
        A sample of the specified uniform symmetric annulus for each band in the simulation
    """
    draw = random.uniform(r1, r2) * random.choice([-1.0, 1.0])
    return [draw] * len(bands.split(','))

## Grid sampling distributions

def poisson_noise(shape, mean):
    """
    Return a grid of values sampled form a Poisson distribution with specifed mean

    Args:
        shape (List[int] or int): Automatically passed based on image shape
        mean: The mean value of the Poisson distirbution to sample from

    Returns:
        A grid of values sampled form a Poisson distribution with specifed mean
    """
    return poisson.rvs(mu=mean, size=shape)


## Empirical distributions from astronomical surveys

# DES
def des_magnitude_zero_point(bands=''):
    """
    Sample from the distribution of single epoch zeropoints for DES
    """
    dist = {'g': 26.58, 'r': 26.78, 'i': 26.75, 'z': 26.48, 'Y': 25.40}
    return [dist[b] for b in bands.split(',')]
    
def des_sky_brightness(bands=''):
    """
    Sample from the distribution of single epoch sky brightness for DES
    """
    # Figure 4 in https://arxiv.org/pdf/1801.03181.pdf
    dist = {'g': {'VALUES': [21.016, 21.057, 21.106, 21.179, 21.228, 21.269, 21.326, 
                             21.367, 21.424, 21.465, 21.522, 21.571, 21.62, 21.677, 
                             21.717, 21.774, 21.823, 21.872, 21.921, 21.97, 22.019, 
                             22.068, 22.117, 22.174, 22.215, 22.272, 22.321, 22.378, 
                             22.427, 22.476],
                  'WEIGHTS': [0.0, 0.0, 0.001, 0.001, 0.001, 0.001, 0.002, 0.003, 
                              0.005, 0.007, 0.009, 0.012, 0.016, 0.023, 0.034, 0.048, 
                              0.063, 0.073, 0.081, 0.093, 0.107, 0.099, 0.087, 0.076, 
                              0.061, 0.05, 0.027, 0.013, 0.005, 0.0]},
            'r': {'VALUES': [20.16, 20.209, 20.266, 20.323, 20.372, 20.421, 20.47, 
                             20.519, 20.576, 20.625, 20.674, 20.715, 20.772, 20.821, 
                             20.87, 20.918, 20.976, 21.024, 21.073, 21.122, 21.171, 
                             21.22, 21.269, 21.326, 21.375, 21.424, 21.473, 21.522, 
                             21.571, 21.62, 21.668, 21.726],
                  'WEIGHTS': [0.0, 0.0, 0.001, 0.001, 0.002, 0.002, 0.005, 0.008, 
                              0.011, 0.011, 0.012, 0.02, 0.023, 0.034, 0.043, 0.046, 
                              0.056, 0.07, 0.075, 0.083, 0.093, 0.095, 0.092, 0.078, 
                              0.057, 0.041, 0.024, 0.012, 0.004, 0.001, 0.0, 0.0]},
            'i': {'VALUES': [18.921, 18.978, 19.027, 19.076, 19.125, 19.174, 19.223, 
                             19.272, 19.321, 19.378, 19.418, 19.476, 19.524, 19.573, 
                             19.622, 19.671, 19.728, 19.777, 19.826, 19.875, 19.924, 
                             19.973, 20.022, 20.071, 20.12, 20.177, 20.226, 20.274, 
                             20.323, 20.372, 20.421, 20.478, 20.527, 20.576, 20.617, 
                             20.674, 20.723, 20.772, 20.829],
                  'WEIGHTS': [0.0, 0.0, 0.002, 0.002, 0.001, 0.002, 0.003, 0.005, 
                              0.013, 0.017, 0.018, 0.026, 0.029, 0.035, 0.036, 0.047, 
                              0.053, 0.067, 0.078, 0.084, 0.073, 0.073, 0.063, 0.05, 
                              0.045, 0.039, 0.031, 0.026, 0.021, 0.018, 0.014, 0.009, 
                              0.009, 0.003, 0.002, 0.002, 0.001, 0.0, 0.0]},
            'z': {'VALUES': [17.715, 17.772, 17.804, 17.861, 17.918, 17.976, 18.024, 
                             18.073, 18.122, 18.171, 18.228, 18.277, 18.326, 18.375, 
                             18.424, 18.473, 18.522, 18.579, 18.628, 18.677, 18.726, 
                             18.774, 18.823, 18.872, 18.921, 18.97, 19.019, 19.076, 
                             19.125, 19.174, 19.231, 19.264, 19.329, 19.37, 19.427, 
                             19.467, 19.524, 19.573, 19.63],
                  'WEIGHTS': [0.0, 0.0, 0.0, 0.001, 0.001, 0.004, 0.007, 0.008, 
                              0.012, 0.014, 0.015, 0.022, 0.028, 0.028, 0.033, 0.045, 
                              0.052, 0.058, 0.064, 0.073, 0.082, 0.078, 0.069, 0.059, 
                              0.051, 0.044, 0.036, 0.024, 0.019, 0.018, 0.017, 0.015, 
                              0.01, 0.005, 0.002, 0.002, 0.002, 0.001, 0.0]},
            'Y': {'VALUES': [17.062, 17.128, 17.177, 17.226, 17.274, 17.323, 17.372, 
                             17.421, 17.47, 17.527, 17.576, 17.625, 17.674, 17.723, 
                             17.772, 17.821, 17.878, 17.927, 17.976, 18.024, 18.073, 
                             18.13, 18.179, 18.228, 18.277, 18.326, 18.375, 18.424, 
                             18.473, 18.53, 18.579, 18.628, 18.668, 18.726, 18.774, 
                             18.823, 18.88, 18.929, 18.97, 19.027, 19.076],
                  'WEIGHTS': [0.001, 0.002, 0.002, 0.003, 0.006, 0.008, 0.011, 0.015, 
                              0.02, 0.027, 0.032, 0.041, 0.051, 0.051, 0.05, 0.05, 
                              0.056, 0.066, 0.072, 0.068, 0.056, 0.047, 0.042, 0.033, 
                              0.032, 0.029, 0.024, 0.022, 0.021, 0.02, 0.014, 0.011, 
                              0.006, 0.003, 0.002, 0.001, 0.001, 0.0, 0.002, 0.001, 0.0]}
            }
    return [random.choices(dist[b]['VALUES'], dist[b]['WEIGHTS'])[0] for b in bands.split(',')]


def des_exposure_time(bands=''):
    """
    Sample from the single epoch exposure time for DES
    """
    # https://arxiv.org/pdf/1801.03181.pdf
    return [45.0 if b == 'Y' else 90.0 for b in bands.split(',')]

def des_seeing(bands=''):
    """
    Sample from the single epoch seeing for DES
    """
    #Figure 3 in https://arxiv.org/pdf/1801.03181.pdf
    dist = {'g': {'VALUES': [0.56, 0.579, 0.601, 0.621, 0.642, 0.662, 0.679, 0.703, 0.72,
                             0.742, 0.761, 0.783, 0.822, 0.841, 0.863, 0.882, 0.902, 0.921,
                             0.943, 0.962, 0.982, 1.001, 1.021, 1.04, 1.062, 1.081, 1.101,
                             1.122, 1.139, 1.161, 1.181, 1.2, 1.219, 1.241, 1.261, 1.282,
                             1.302, 1.319, 1.341, 1.36, 1.379, 1.399, 1.418, 1.44, 1.479,
                             1.501, 1.52, 1.539, 1.559, 1.578, 1.598, 1.619, 1.639, 1.658,
                             1.678, 1.697, 1.719, 1.738, 1.758, 1.777, 1.799, 1.82],
                  'WEIGHTS': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.001, 0.002,
                              0.004, 0.005, 0.008, 0.011, 0.015, 0.02, 0.025, 0.029, 0.034, 
			      0.038, 0.042, 0.045, 0.046, 0.047, 0.045, 0.044, 0.041, 0.04, 
                              0.037, 0.033, 0.031, 0.028, 0.026, 0.025, 0.023, 0.021, 0.019, 
                              0.018, 0.017, 0.016, 0.015, 0.014, 0.014, 0.013, 0.012, 0.012, 
                              0.011, 0.01, 0.009, 0.009, 0.008, 0.007, 0.007, 0.006, 0.005, 
                              0.004, 0.003, 0.002, 0.001, 0.001, 0.0]},
            'r': {'VALUES': [0.56, 0.579, 0.601, 0.621, 0.642, 0.662, 0.679, 0.703, 0.72, 
                             0.742, 0.761, 0.783, 0.822, 0.841, 0.863, 0.882, 0.902, 0.921, 
                             0.943, 0.962, 0.982, 1.001, 1.021, 1.04, 1.062, 1.081, 1.101, 
                             1.122, 1.139, 1.161, 1.181, 1.2, 1.219, 1.241, 1.261, 1.282, 
                             1.302, 1.319, 1.341, 1.36, 1.379, 1.399, 1.418, 1.44, 1.479, 
                             1.501, 1.52, 1.539, 1.559, 1.578, 1.598, 1.619, 1.639, 1.658, 
                             1.678, 1.697, 1.719, 1.738, 1.758, 1.777, 1.799, 1.82],
                  'WEIGHTS': [0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.002, 0.004, 0.007, 0.012, 
                              0.019, 0.027, 0.036, 0.043, 0.051, 0.057, 0.062, 0.063, 0.061, 
                              0.058, 0.054, 0.048, 0.044, 0.04, 0.036, 0.032, 0.028, 0.025, 
                              0.022, 0.019, 0.018, 0.015, 0.014, 0.012, 0.011, 0.01, 0.009, 
                              0.008, 0.008, 0.007, 0.006, 0.005, 0.004, 0.004, 0.003, 0.003, 
                              0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.0, 0.0, 
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            'i': {'VALUES': [0.56, 0.579, 0.601, 0.621, 0.642, 0.662, 0.679, 0.703, 0.72, 
                             0.742, 0.761, 0.783, 0.822, 0.841, 0.863, 0.882, 0.902, 0.921, 
                             0.943, 0.962, 0.982, 1.001, 1.021, 1.04, 1.062, 1.081, 1.101, 
                             1.122, 1.139, 1.161, 1.181, 1.2, 1.219, 1.241, 1.261, 1.282, 
                             1.302, 1.319, 1.341, 1.36, 1.379, 1.399, 1.418, 1.44, 1.479, 
                             1.501, 1.52, 1.539, 1.559, 1.578, 1.598, 1.619, 1.639, 1.658, 
                             1.678, 1.697, 1.719, 1.738, 1.758, 1.777, 1.799, 1.82],
                  'WEIGHTS': [0.0, 0.0, 0.0, 0.001, 0.002, 0.005, 0.01, 0.017, 0.027, 0.038, 
                              0.049, 0.061, 0.067, 0.072, 0.076, 0.075, 0.071, 0.066, 0.058, 
                              0.05, 0.045, 0.038, 0.032, 0.026, 0.021, 0.017, 0.014, 0.011, 
                              0.009, 0.008, 0.007, 0.005, 0.004, 0.003, 0.003, 0.002, 0.002, 
                              0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0, 0.0, 
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                              0.0, 0.0, 0.0, 0.0]},
            'z': {'VALUES': [0.56, 0.579, 0.601, 0.621, 0.642, 0.662, 0.679, 0.703, 0.72, 
                             0.742, 0.761, 0.783, 0.822, 0.841, 0.863, 0.882, 0.902, 0.921, 
                             0.943, 0.962, 0.982, 1.001, 1.021, 1.04, 1.062, 1.081, 1.101, 
                             1.122, 1.139, 1.161, 1.181, 1.2, 1.219, 1.241, 1.261, 1.282, 
                             1.302, 1.319, 1.341, 1.36, 1.379, 1.399, 1.418, 1.44, 1.479, 
                             1.501, 1.52, 1.539, 1.559, 1.578, 1.598, 1.619, 1.639, 1.658, 
                             1.678, 1.697, 1.719, 1.738, 1.758, 1.777, 1.799, 1.82],
                  'WEIGHTS': [0.0, 0.0, 0.001, 0.003, 0.008, 0.016, 0.027, 0.039, 0.054, 
                              0.066, 0.073, 0.077, 0.077, 0.073, 0.069, 0.061, 0.054, 0.045, 
                              0.037, 0.032, 0.026, 0.022, 0.019, 0.017, 0.014, 0.013, 0.011, 
                              0.009, 0.008, 0.007, 0.007, 0.005, 0.004, 0.003, 0.003, 0.003, 
                              0.003, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 
                              0.001, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            'Y': {'VALUES': [0.56, 0.579, 0.601, 0.621, 0.642, 0.662, 0.679, 0.703, 0.72, 
                             0.742, 0.761, 0.783, 0.822, 0.841, 0.863, 0.882, 0.902, 0.921, 
                             0.943, 0.962, 0.982, 1.001, 1.021, 1.04, 1.062, 1.081, 1.101, 
                             1.122, 1.139, 1.161, 1.181, 1.2, 1.219, 1.241, 1.261, 1.282, 
                             1.302, 1.319, 1.341, 1.36, 1.379, 1.399, 1.418, 1.44, 1.479, 
                             1.501, 1.52, 1.539, 1.559, 1.578, 1.598, 1.619, 1.639, 1.658, 
                             1.678, 1.697, 1.719, 1.738, 1.758, 1.777, 1.799, 1.82],
                  'WEIGHTS': [0.0, 0.001, 0.001, 0.004, 0.008, 0.014, 0.023, 0.032, 0.038, 
                              0.045, 0.049, 0.051, 0.051, 0.048, 0.046, 0.043, 0.039, 0.036, 
                              0.033, 0.031, 0.028, 0.026, 0.023, 0.021, 0.019, 0.018, 0.017, 
                              0.016, 0.016, 0.014, 0.013, 0.013, 0.012, 0.011, 0.01, 0.01, 
                              0.01, 0.01, 0.009, 0.008, 0.008, 0.007, 0.008, 0.008, 0.007, 
                              0.007, 0.007, 0.006, 0.006, 0.006, 0.006, 0.006, 0.005, 0.004, 
                              0.003, 0.003, 0.002, 0.002, 0.001, 0.001, 0.0, 0.0]}
            }
    return [random.choices(dist[b]['VALUES'], dist[b]['WEIGHTS'])[0] for b in bands.split(',')]

def des_ccd_gain(bands=''):
    """
    Sample from the single epoch ccd gain for DECam
    """
    # Figure 2 in https://arxiv.org/pdf/1501.02802.pdf
    return [5.033 if b == 'Y' else 6.083 for b in bands.split(',')]

def des_num_exposures(bands=''):
    """
    Sample from the effective number of exposures for DES
    """
    # Figure 5 in https://arxiv.org/pdf/1501.02802.pdf
    dist = {'g': {'VALUES': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                  'WEIGHTS': [0.040, 0.113, 0.267, 0.311, 0.178, 0.062, 0.019, 0.007, 0.003]},
            'r': {'VALUES': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                  'WEIGHTS': [0.041, 0.119, 0.284, 0.321, 0.167, 0.046, 0.014, 0.006, 0.002]},
            'i': {'VALUES': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                  'WEIGHTS': [0.043, 0.121, 0.291, 0.334, 0.165, 0.033, 0.009, 0.003, 0.001]},
            'z': {'VALUES': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                  'WEIGHTS': [0.039, 0.106, 0.272, 0.332, 0.183, 0.048, 0.013, 0.005, 0.002]},
            'Y': {'VALUES': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                  'WEIGHTS': [0.034, 0.074, 0.195, 0.305, 0.241, 0.099, 0.035, 0.012, 0.005]}
            }
    return [random.choices(dist[b]['VALUES'], dist[b]['WEIGHTS'])[0] for b in bands.split(',')]


# DELVE
def delve_seeing(bands=''):
    """
    Sample from the seeing distribution for DELVE observations
    """
    # Erik Zaborowski and Alex Drlica-Wagner
    dist = {'g': {'VALUES': [0.036, 0.107, 0.178, 0.249, 0.32, 0.392, 0.463, 0.534, 0.605, 0.676,
                             0.748, 0.819, 0.89, 0.961, 1.032, 1.104, 1.175, 1.246, 1.317, 1.388,
                             1.46, 1.531, 1.602, 1.673, 1.744, 1.816, 1.887, 1.958, 2.029, 2.1],
                  'WEIGHTS': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002, 0.019,
                              0.044, 0.076, 0.107, 0.137, 0.119, 0.126, 0.104, 0.089, 0.067, 0.049,
                              0.026, 0.017, 0.01, 0.003, 0.003, 0.001, 0.001]},
            'r': {'VALUES': [0.811, 0.85, 0.889, 0.928, 0.967, 1.007, 1.046, 1.085, 1.124, 1.163,
                             1.203, 1.242, 1.281, 1.32, 1.359, 1.398, 1.438, 1.477, 1.516, 1.555,
                             1.594, 1.634, 1.673, 1.712, 1.751, 1.79, 1.829, 1.869, 1.908, 1.947],
                  'WEIGHTS': [0.005, 0.011, 0.024, 0.039, 0.062, 0.075, 0.079, 0.078, 0.082, 0.081,
                              0.065, 0.063, 0.055, 0.052, 0.045, 0.04, 0.033, 0.029, 0.023, 0.016,
                              0.013, 0.01, 0.005, 0.005, 0.002, 0.002, 0.003, 0.001, 0.001, 0.001]},
            'i': {'VALUES': [0.745, 0.792, 0.839, 0.887, 0.934, 0.981, 1.028, 1.075, 1.123, 1.17,
                             1.217, 1.264, 1.312, 1.359, 1.406, 1.453, 1.5, 1.548, 1.595, 1.642, 1.689,
                             1.737, 1.784, 1.831, 1.878, 1.925, 1.973, 2.02, 2.067, 2.114],
                  'WEIGHTS': [0.003, 0.015, 0.033, 0.06, 0.088, 0.126, 0.127, 0.123, 0.094, 0.072, 0.058,
                              0.052, 0.035, 0.028, 0.023, 0.018, 0.01, 0.009, 0.007, 0.006, 0.004, 0.002,
                              0.002, 0.001, 0.002, 0.001, 0.0, 0.001, 0.0, 0.0]},
            'z': {'VALUES': [0.754, 0.801, 0.849, 0.896, 0.944, 0.991, 1.039, 1.086, 1.134, 1.181, 1.229,
                             1.276, 1.324, 1.371, 1.419, 1.466, 1.514, 1.561, 1.609, 1.656, 1.704, 1.751,
                             1.799, 1.846, 1.894, 1.941, 1.989, 2.036, 2.084, 2.131],
                  'WEIGHTS': [0.009, 0.036, 0.082, 0.111, 0.105, 0.104, 0.091, 0.085, 0.087, 0.065, 0.051,
                              0.045, 0.035, 0.023, 0.022, 0.013, 0.009, 0.009, 0.005, 0.004, 0.003, 0.002,
                              0.001, 0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001]}
            }
    return [random.choices(dist[b]['VALUES'], dist[b]['WEIGHTS'])[0] for b in bands.split(',')]

def delve_sky_brightness(bands=''):
    """
    Sample from the sky brightness distribution for DELVE observaitons
    """
    # Erik Zaborowski and Alex Drlica-Wagner
    dist = {'g': {'VALUES': [18.201, 18.362, 18.524, 18.685, 18.847, 19.008, 19.17, 19.331, 19.493, 19.654,
                             19.816, 19.977, 20.138, 20.3, 20.461, 20.623, 20.784, 20.946, 21.107, 21.269,
                             21.43, 21.592, 21.753, 21.915, 22.076, 22.238, 22.399, 22.561, 22.722, 22.884],
                  'WEIGHTS': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001,
                              0.001, 0.003, 0.007, 0.021, 0.024, 0.03, 0.075, 0.202, 0.32, 0.195, 0.099,
                              0.022, 0.0, 0.0, 0.0]},
            'r': {'VALUES': [16.705, 17.076, 17.447, 17.818, 18.189, 18.56, 18.931, 19.302, 19.674, 20.045,
                             20.416, 20.787, 21.158, 21.529, 21.9, 22.271, 22.642, 23.013, 23.384, 23.755,
                             24.126, 24.497, 24.868, 25.239, 25.61, 25.982, 26.353, 26.724, 27.095, 27.466],
                  'WEIGHTS': [0.0, 0.0, 0.0, 0.001, 0.0, 0.001, 0.001, 0.001, 0.002, 0.024, 0.084, 0.312,
                              0.411, 0.157, 0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.001, 0.001]},
            'i': {'VALUES': [16.481, 16.63, 16.778, 16.927, 17.075, 17.224, 17.373, 17.521, 17.67, 17.818,
                             17.967, 18.116, 18.264, 18.413, 18.561, 18.71, 18.859, 19.007, 19.156, 19.304,
                             19.453, 19.602, 19.75, 19.899, 20.047, 20.196, 20.344, 20.493, 20.642, 20.79],
                  'WEIGHTS': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002, 0.003,
                              0.009, 0.023, 0.039, 0.057, 0.1, 0.12, 0.137, 0.138, 0.128, 0.091, 0.068, 0.05,
                              0.029, 0.005, 0.001]},
            'z': {'VALUES': [13.205, 13.604, 14.004, 14.404, 14.804, 15.203, 15.603, 16.003, 16.403, 16.802,
                             17.202, 17.602, 18.001, 18.401, 18.801, 19.201, 19.6, 20.0, 20.4, 20.8, 21.199,
                             21.599, 21.999, 22.398, 22.798, 23.198, 23.598, 23.997, 24.397, 24.797],
                  'WEIGHTS': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.001, 0.001, 0.001, 0.002, 0.009, 0.043,
                              0.237, 0.452, 0.2, 0.038, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.005,
                              0.007, 0.002, 0.0]}
            }

    return [random.choices(dist[b]['VALUES'], dist[b]['WEIGHTS'])[0] for b in bands.split(',')]

def delve_exposure_time(bands=''):
    """
    Sample from the exposure time distribtuion for DELVE observations
    """
    # Erik Zaborowski and Alex Drlica-Wagner
    dist = {'g': {'VALUES': [35.333, 46.0, 56.667, 67.333, 78.0, 88.667, 99.333,
			     110.0, 120.667, 131.333, 142.0, 152.667, 163.333, 174.0,
			     184.667, 195.333, 206.0, 216.667, 227.333, 238.0, 248.667,
			     259.333, 270.0, 280.667, 291.333, 302.0, 312.667, 323.333,
			     334.0, 344.667],
	          'WEIGHTS': [0.091, 0.031, 0.06, 0.024, 0.021, 0.4, 0.108, 0.018, 0.041,
	   		      0.015, 0.004, 0.003, 0.029, 0.015, 0.004, 0.024, 0.003, 0.0,
	   		      0.0, 0.012, 0.013, 0.003, 0.026, 0.008, 0.0, 0.044, 0.0, 0.0,
	   		      0.0, 0.003]},
            'r': {'VALUES': [35.333, 46.0, 56.667, 67.333, 78.0, 88.667, 99.333, 110.0,
 			     120.667, 131.333, 142.0, 152.667, 163.333, 174.0, 184.667,
 			     195.333, 206.0, 216.667, 227.333, 238.0, 248.667, 259.333,
 			     270.0, 280.667, 291.333, 302.0, 312.667, 323.333, 334.0, 344.667],
 	          'WEIGHTS': [0.294, 0.069, 0.15, 0.033, 0.03, 0.085, 0.113, 0.007, 0.03,
 	                      0.001, 0.0, 0.085, 0.004, 0.004, 0.019, 0.02, 0.006, 0.0, 0.007,
 	                      0.0, 0.01, 0.001, 0.018, 0.0, 0.0, 0.012, 0.0, 0.0, 0.0, 0.002]},
            'i': {'VALUES': [35.333, 46.0, 56.667, 67.333, 78.0, 88.667, 99.333, 110.0, 120.667,
                             131.333, 142.0, 152.667, 163.333, 174.0, 184.667, 195.333, 206.0,
                             216.667, 227.333, 238.0, 248.667, 259.333, 270.0, 280.667, 291.333,
                             302.0, 312.667, 323.333, 334.0, 344.667],
                  'WEIGHTS': [0.275, 0.029, 0.064, 0.048, 0.045, 0.241, 0.044, 0.007, 0.05, 0.007,
                              0.014, 0.042, 0.018, 0.0, 0.012, 0.02, 0.005, 0.005, 0.0, 0.0, 0.042,
                              0.0, 0.0, 0.0, 0.0, 0.021, 0.0, 0.0, 0.009, 0.002]},
            'z': {'VALUES': [35.05, 45.15, 55.25, 65.35, 75.45, 85.55, 95.65, 105.75, 115.85,
                             125.95, 136.05, 146.15, 156.25, 166.35, 176.45, 186.55, 196.65,
                             206.75, 216.85, 226.95, 237.05, 247.15, 257.25, 267.35, 277.45,
                             287.55, 297.65, 307.75, 317.85, 327.95],
                  'WEIGHTS': [0.128, 0.043, 0.152, 0.062, 0.083, 0.162, 0.107, 0.022, 0.06, 0.015,
                              0.013, 0.019, 0.019, 0.006, 0.005, 0.005, 0.01, 0.002, 0.003, 0.001,
                              0.028, 0.011, 0.0, 0.01, 0.009, 0.0, 0.016, 0.0, 0.0, 0.009]}
            }
    return [random.choices(dist[b]['VALUES'], dist[b]['WEIGHTS'])[0] for b in bands.split(',')]

def delve_magnitude_zero_point(bands=''):
    """
    Sample from the zero point distribtutions for DELVE observaitons
    """
    # Erik Zaborowski and Alex Drlica-Wagner
    dist = {'g': 31.550, 'r': 31.284, 'i': 31.608, 'z': 31.262}
    return [dist[b] for b in bands.split(',')]

# LSST at the Vera C. Rubin Observatory
def lsst_num_exposures(bands='', coadd_years=10):
    """
    Sample from the LSST number of exposures distribution

    Args:
        coadd_years (int): Number of years of the survey to utlize
    """
    dist = {'u': 140, 'g': 200, 'r': 460, 'i': 460, 'z': 400, 'Y': 400}
    return [coadd_years * dist[b] // 10 for b in bands.split(',')]

def lsst_exposure_time(bands=''):
    """
    Sample from the LSST exposure time distribution
    """
    dist = {'u': 15.0, 'g': 15.0, 'r': 15.0, 'i': 15.0, 'z': 15.0, 'Y': 15.0}
    return [dist[b] for b in bands.split(',')]

def lsst_magnitude_zero_point(bands=''):
    """
    Sample from the LSST zero point distribution
    """
    dist = {'u': 26.5, 'g': 28.3, 'r': 28.13, 'i': 27.79, 'z': 27.40, 'Y': 26.58}
    return [dist[b] for b in bands.split(',')]

def lsst_sky_brightness(bands=''):
    """
    Sample from the LSST sky brightness distribution
    """
    dist = {'u': 22.99, 'g': 22.26, 'r': 21.2, 'i': 20.48, 'z': 19.6, 'Y': 18.61}
    return [dist[b] for b in bands.split(',')]

def lsst_seeing(bands=''):
    """
    Sample from the LSST seeing distribution
    """
    dist = {'u': 0.81, 'g': 0.77, 'r': 0.73, 'i': 0.71, 'z': 0.69, 'Y': 0.68}
    return [dist[b] for b in bands.split(',')]

# ZTF
def ztf_magnitude_zero_point(bands=''):
    """
    Sample from the ZTF zeropoint distribution
    """
    dist = {'g': 26.325, 'r': 26.275, 'i': 25.660}
    return [dist[b] for b in bands.split(',')]

def ztf_seeing(bands=''):
    """
    Sample from the ZTF seeing distribution
    """
    dist = {'g': 2.1, 'r': 2.0, 'i': 2.1}
    return [dist[b] for b in bands.split(',')]

def ztf_sky_brightness(bands=''):
    """
    Sample from the ZTF sky brightness distribution
    """
    dist = {'g': 22.01, 'r': 21.15, 'i': 19.89}
    return [dist[b] for b in bands.split(',')]
    

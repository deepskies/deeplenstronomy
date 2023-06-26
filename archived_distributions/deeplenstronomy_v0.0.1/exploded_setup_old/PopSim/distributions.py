# A module listing all available distribtuions

import numpy as np
import random

def uniform(minimum, maximum, bands=''):
    draw = random.uniform(minimum, maximum)
    return [draw] * len(bands.split(','))

def normal(mean, std, bands=''):
    draw = np.random.normal(loc=mean, scale=std)
    return [draw] * len(bands.split(','))

def delta_function(value, bands=''):
    return [value] * len(bands.split(','))
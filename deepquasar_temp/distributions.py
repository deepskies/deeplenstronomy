# A module listing all available distribtuions

import numpy as np
import random

def uniform(minimum, maximum):
    return random.uniform(minimum, maximum)

def normal(mean, std):
    return np.random.normal(loc=mean, scale=std)

def delta_function(value):
    return value
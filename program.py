# -*- coding: utf-8 -*-
"""
@author: Parker Hutchinson
"""

import numpy as np
import tensorflow as tf
import pandas as pd

import random

def noise(variance):
    return random.randrange(-variance, variance)
    
def target_function(x):
    return 2 * x + noise(2)

data = {'X':[], 'Y':[]}
for i in range(0, 1000):
    data['X'].append(i)
    data['Y'].append(target_function(i))

df = pd.DataFrame(data)



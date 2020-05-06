import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import poisson, halfnorm
from scipy.ndimage import gaussian_filter
from copy import copy
import multiprocessing 
import pandas as pd
from matplotlib import rc, cm
from itertools import product
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

"""
General Helpers
"""
## Round to nearest value on axis
def fn(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

def resample(x,w,N):
    if x.shape[0] == 2:
        ix = np.random.choice(np.arange(x.shape[1]), p=w, size=N)
        x_r = x[:,ix]
    else:
        x_r = np.random.choice(x, p=w, size=N)
    w_r = np.ones(N) * (1.0/N)
    return x_r, w_r



import numpy as np
import pandas as pd
from helper_functions import fn
from scipy.ndimage import gaussian_filter

def poisson_mean(spikes, i, x_k, X, time_bucket_length):
    if len(np.where(X==x_k)[0]) > 0:
        return spikes[np.where(X==x_k),i].sum()/(len(np.where(X==x_k)[0]) * time_bucket_length)
    return 0

def calculate_poisson(spikes, X, xaxis, time_bucket_length):
    poisson = {}
    num_neurons = spikes.shape[1]
    for x_k in xaxis:
        for i in range(num_neurons):
            pm = poisson_mean(spikes, i, x_k, X, time_bucket_length)
            poisson[i, x_k] = pm
    return poisson

def estimate_place_fields(spikes, X, xaxis, dt, sigma=40):
    find_nearest = np.vectorize(lambda x: fn(xaxis, x))
    X_copy = find_nearest(np.copy(X))
    poisson_dict = calculate_poisson(spikes, X_copy, xaxis, dt)
    poisson_series = pd.Series(poisson_dict).reset_index()
    poisson_series.columns = ['neuron', 'location', 'poisson_mean']
    poisson_series = poisson_series.sort_values(['neuron', 'location']).reset_index(drop=True)
    estimated_place_fields = np.array(poisson_series.pivot(index='neuron',columns='location'))
    smoothed_place_fields = np.apply_along_axis(lambda x: gaussian_filter(x, sigma=sigma, mode='nearest'), 1, estimated_place_fields)
    return estimated_place_fields, smoothed_place_fields
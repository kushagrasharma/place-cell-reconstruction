import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import poisson, halfnorm
from scipy.ndimage import gaussian_filter
from copy import copy
import multiprocessing 
import pickle
import pandas as pd
from itertools import product

## Round to nearest value on axis
def fn(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]
    
def generatePlaceFields1D(N,xaxis,L):
    # N: number of neurons
    # L: length of the linear track
    # dL: spatial discretization
    
    
    L = abs(xaxis[-1]-xaxis[0])
    dL = xaxis[1]-xaxis[0]
    f = np.zeros((N,xaxis.size))
    for i in range(N):
        # draw place field parameters randomly
        m = np.random.poisson(1) + 1   # number of place fields
        mu = np.random.uniform(xaxis.min(),xaxis.max(),m)  # place field centers
        sigma = np.random.uniform(dL,L/4,m) # place field width
        f0 = np.random.uniform(0,10,m) # maximal firing rate
        for j in range(m):
            f[i] = f[i] + f0[j] * np.exp(-(xaxis-mu[j])**2/sigma[j]**2)
    
    return f

def h(x,f,xaxis,dL):
# evaluates the place fields on a grid
# returns spiking frequencies at respective locations from x
    bins = xaxis-dL/2
    ind = np.digitize(x,bins) - 1
    hx = f[:,ind]
    
    return hx
    
    
def poisson_mean(spikes, i, x_k, x, time_bucket_length):
    if len(np.where(x==x_k)[0]) > 0:
        return spikes[np.where(x==x_k),i].sum()/(len(np.where(x==x_k)[0]) * time_bucket_length)
    return 0

def calculate_poisson(spikes, x, xaxis, time_bucket_length):
    poisson = {}
    num_neurons = spikes.shape[1]
    for x_k in xaxis:
        for i in range(num_neurons):
            pm = poisson_mean(spikes, i, x_k, x, time_bucket_length)
            poisson[i, x_k] = pm
    return poisson

def estimate_place_fields(spikes, X, xaxis, dt, sigma=40):
    poisson_dict = calculate_poisson(spikes, X, xaxis, dt)
    poisson_series = pd.Series(poisson_dict).reset_index()
    poisson_series.columns = ['neuron', 'location', 'poisson_mean']
    poisson_series = poisson_series.sort_values(['neuron', 'location']).reset_index(drop=True)
    estimated_place_fields = np.array(poisson_series.pivot(index='neuron',columns='location'))
    smoothed_place_fields = np.apply_along_axis(lambda x: gaussian_filter(x, sigma=sigma), 1, estimated_place_fields)
    return estimated_place_fields, smoothed_place_fields

def resample(x,w,N):
    if x.shape[0] == 2:
        ix = np.random.choice(np.arange(x.shape[1]), p=w, size=N)
        x_r = x[:,ix]
    else:
        x_r = np.random.choice(x, p=w, size=N)
    w_r = np.ones(N) * (1.0/N)
    return x_r,w_r

T = 150
dt = 0.01
t = np.arange(0,T,dt)
M = 10 # number of neurons
L = 250 # total length
dL = 0.1
P= 1000
Peff_min = 0.5
xaxis = np.arange(0,L,dL) + dL/2 # center of bins

params = {
    "tau": 10,
    "q": 100,
    "dt": dt,
    "xaxis": xaxis, # axis for placefields
    "dL": dL,
    "T": T,
    "P": P,
    "Peff_min": Peff_min,
    "L": L
}

X = np.zeros((int(T/dt),2)) # x and v components
X[0,0] = 10
X[0,1] = 0 
for i in np.arange(1,int(T/dt)):
    x = np.copy(X[i-1,0])
    v = np.copy(X[i-1,1])
    # integrate one timestep using Euler scheme
    X[i,0] = x + v * dt
    X[i,1] = v - 1/params["tau"] * v * dt + params["q"] * np.random.normal(0,np.sqrt(dt))
    # implement Skohorod boundary conditions
    if X[i,0] < 0:
        X[i,0] = 0 # put particle to boundary
        X[i,1] = 0 # set velocity to zero
    elif X[i,0] > L:
        X[i,0] = L # put particle to boundary
        X[i,1] = 0 # set velocity to zero

find_nearest = np.vectorize(lambda x: fn(xaxis, x))
X[:,0] = find_nearest(X[:,0])
        
f = generatePlaceFields1D(M,xaxis,L)
params["f"] = f
H = h(X[:,0],f,xaxis,dL)
dN = np.random.poisson(H * dt)
spikes = dN.transpose()
X[:,0] = np.around(X[:,0],2)
xaxis = np.around(xaxis,2)
X_train, X_validation, X_test = np.split(X, [int(.6*len(X)), int(.8*len(X))])
spikes_train, spikes_validation, spikes_test = np.split(spikes, [int(.6*len(spikes)), int(.8*len(spikes))])
estimated_place_fields, smoothed_place_fields = estimate_place_fields(spikes_train, X_train[:,0], xaxis, dt)
params["estimated_place_fields"] = estimated_place_fields
params["smoothed_place_fields"] = smoothed_place_fields

def randomwalk_pf(spikes, params):
    T = params["T"]
    dt = params["dt"]
    P = params["P"]
    tau = params["tau"]
    Peff_min = params["Peff_min"]
    L = params["L"]
    xaxis = params["xaxis"]
    place_fields = params["smoothed_place_fields"]
    dL = params["dL"]
    q = params["q"]
    t = int(T/dt)
    
    x = np.zeros((t, 2, P))
    w = np.full((t, P), 1/P)
    
    def boundary(x_t):
        if x_t[0] < 0:
            x_t[0] = 0
            x_t[1] = 0
        elif x_t[0] > L:
            x_t[0] = L
            x_t[1] = 0
        return x_t
    
    for i in range(1, t):
        # particle transition
        x[i,0] = x[i-1,0] + x[i-1,1] * dt
        dW = np.random.normal(scale=np.sqrt(dt), size=P)
        x[i,1] = x[i-1,1] - (x[i-1,1] / tau) * dt + q * dW
        x[i] = np.apply_along_axis(boundary, 0, x[i])
        
        # weight update
        w[i] = w[i-1] * np.prod( poisson.pmf(spikes[i],h(x[i,0],place_fields,xaxis,dL).transpose()*dt) , 1)

        w[i] /= np.sum(w[i])
        
        while np.isnan(np.sum(w[i])):
            # all particles have 0 posterior probability, so we need to resample from the uniform
            x[i,0] = np.random.uniform(0, L, P)
            x[i,1] = 0
            w[i] = (1/P) * np.prod( poisson.pmf(spikes[i],h(x[i,0],place_fields,xaxis,dL).transpose()*dt) , 1)
            w[i] /= np.sum(w[i])
        
        if 1 / np.sum(w[i] ** 2) < Peff_min * P:
            x[i], w[i] = resample(x[i], w[i], P)
                   
    return x, w

def get_error_for_params(X, spikes, q, tau):
    params["q"] = q
    params["tau"] = tau
    x_pf, w_pf = randomwalk_pf(spikes, params)
    mu = np.sum(x_pf[:,0,:] * w_pf, 1)
    mean_validation_err = abs(mu-X[:,0]).mean()
    return q, tau, mean_validation_err 

def get_error_for_params_with_range(q, tau):
    return get_error_for_params(X_validation, spikes_validation, q, tau)

qs = [0.1, 0.5, 1, 2, 5, 10, 15, 20, 50, 100, 500, 1000]
taus = [0.1, 0.5, 1, 2, 5, 10, 15, 20, 50, 100, 500, 1000]
validation_errors = []

params["T"] = int(len(X_validation) * dt)
pool = multiprocessing.Pool()
validation_errors = pool.starmap(get_error_for_params_with_range, product(qs, taus))

pickle.dump( validation_errors, open( "validation_errors.json", "wb" ) )

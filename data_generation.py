import numpy as np
from helper_functions import fn

""" 
Trajectory Generation
"""
def generate_random_walk_trajectory(params, boundaries=True, round_data=False):
    T = params["T"]
    dt = params["dt"]
    L = params["L"] if "L" in params else None
    tau = params["tau"]
    q = params["q"]
    dL = params["dL"]
    
    X = np.zeros((int(T/dt),2)) # x and v components
    X[0,0] = 0
    X[0,1] = 0 
    for i in np.arange(1,int(T/dt)):
        x = np.copy(X[i-1,0])
        v = np.copy(X[i-1,1])
        # integrate one timestep using Euler scheme
        X[i,0] = x + v * dt
        X[i,1] = v - 1/tau * v * dt + q * np.random.normal(0,np.sqrt(dt))
        # implement Skohorod boundary conditions
        if boundaries:
            if X[i,0] < 0:
                X[i,0] = 0 # put particle to boundary
                X[i,1] = 0 # set velocity to zero
            elif X[i,0] > L:
                X[i,0] = L # put particle to boundary
                X[i,1] = 0 # set velocity to zero
    if boundaries:
        xaxis = np.arange(0,L,dL) + dL/2 # center of bins
    else:
        xaxis = np.arange(int(X[:,0].min()-2),int(X[:,0].max()+2),dL) + dL/2 # center of bins
    if round_data:
        find_nearest = np.vectorize(lambda x: fn(xaxis, x))
        X[:,0] = find_nearest(X[:,0])
    
    return X, xaxis

def generate_diffusion_trajectory(params, boundaries=True, round_data=False):
    T = params["T"]
    dt = params["dt"]
    L = params["L"] if "L" in params else None
    dL = params["dL"]
    
    X = np.zeros(int(T/dt)) 
    X[0] = 0
    
    for i in np.arange(1,int(T/dt)):
        x = np.copy(X[i-1])
        # integrate one timestep using Euler scheme
        X[i] = x + params["sigma"] * np.random.normal(0,np.sqrt(dt))
        if boundaries:
            if X[i] < 0:
                X[i] = 0 # put particle to boundary
            elif X[i] > L:
                X[i] = L # put particle to boundary
    if boundaries:
        xaxis = np.arange(0,L,dL) + dL/2 # center of bins
    else:
        xaxis = np.arange(int(X[:,0].min()-2),int(X[:,0].max()+2),dL) + dL/2 # center of bins        
    if round_data:
        find_nearest = np.vectorize(lambda x: fn(xaxis, x))
        X = find_nearest(X)

    return X, xaxis

"""
Place Field Generation
"""

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

def h(X,f,xaxis,dL):
# evaluates the place fields on a grid
# returns spiking frequencies at respective locations from x
    find_nearest = np.vectorize(lambda x: fn(xaxis, x))
    X = find_nearest(X)
    bins = xaxis-dL/2
    ind = np.digitize(X,bins) - 1
    hx = f[:,ind]
    
    return hx
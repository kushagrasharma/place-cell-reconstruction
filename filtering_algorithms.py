import numpy as np
from data_generation import h
from scipy.stats import poisson
from helper_functions import resample

def diffusion_pf(spikes, params):
    T = params["T"]
    dt = params["dt"]
    P = params["P"]
    sigma = params["sigma"]
    Peff_min = params["Peff_min"]
    L = params["L"]
    xaxis = params["xaxis"]
    place_fields = params["smoothed_place_fields"]
    dL = params["dL"]
    t = int(T/dt)
    
    x = np.zeros((t, P))
    w = np.full((t, P), 1/P)

    if "initial_condition" in params:
        x[0,:] = params["initial_condition"]
    
    def boundary(x_t):
        if x_t < 0:
            return 0 # put particle to boundary
        elif x_t > L:
            return L # put particle to boundary
        return x_t
            
    vec_boundary = np.vectorize(boundary)
    
    for i in range(1, t):
        # particle transition
        x[i] = x[i-1] + sigma * np.random.normal(0, np.sqrt(dt), P)
        x[i] = vec_boundary(x[i])
        
        # weight update
        w[i] = w[i-1] * np.prod( poisson.pmf(spikes[i],h(x[i],place_fields,xaxis,dL).transpose()*dt) , 1)

        w[i] /= np.sum(w[i])
        
        while np.isnan(np.sum(w[i])):
            # all particles have 0 posterior probability, so we need to resample from the uniform
            x[i] = np.random.uniform(0, L, P)
            w[i] = (1/P) * np.prod( poisson.pmf(spikes[i],h(x[i],place_fields,xaxis,dL).transpose()*dt) , 1)
            w[i] /= np.sum(w[i])
        
        if 1 / np.sum(w[i] ** 2) < Peff_min * P:
            x[i], w[i] = resample(x[i], w[i], P)
                   
    return x, w

def random_walk_pf(spikes, params):
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

    if "initial_condition" in params:
        x[0,0,:] = params["initial_condition"]
    
    def boundary(x_t):
        if x_t[0] <= 0:
            x_t[0] = 0
            x_t[1] = 0
        elif x_t[0] >= L:
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

import numpy as np
import pandas as pd
from itertools import product

nan = np.nan

def fit_diffusion_model(X, dt):
    if len(X.shape) > 1:
        diffs = np.diff(X[:,0])
    else:
        diffs = np.diff(X)
    analytic_sigma = (1/(X.shape[0] * dt)) * np.sum(diffs ** 2)
    analytic_sigma = np.sqrt(analytic_sigma)
    return analytic_sigma

def get_diffusion_likelihood(X, dt, sigma):
    sd = sigma * np.sqrt(dt)
    if len(X.shape) > 1:
        diffs = np.diff(X[:,0])
    else:
        diffs = np.diff(X)
    l = (1 - X.shape[0]) * np.log(sd) - (1 / (2 * (sd ** 2))) * np.sum(diffs ** 2)
    return -l

def fit_random_walk_model(X, dt, use_velocity=False, q=None, tau=None):
    if len(X.shape) > 1:
        pos = X[:,0]
    else:
        pos = X
    if use_velocity:
        base_velocity = X[:,1]
    else:
        base_velocity = np.diff(pos) / dt
            
    mu_hat = 1 - (np.sum(base_velocity[1:] * base_velocity[:-1]) / np.sum(base_velocity[:-1] ** 2))
    mu_hat /= dt
    
    if tau:
        mu_hat = 1.0 / tau
    
    q_hat = np.sum((base_velocity[1:] - base_velocity[:-1] + base_velocity[:-1] * mu_hat * dt) ** 2)
    q_hat /= dt * (X.shape[0] - 1)
    q_hat = np.sqrt(q_hat)
    
    if q:
        q_hat = q
    
    tau_hat = 1.0 / mu_hat
    
    return q_hat, tau_hat

def using_clump(a):
    return [a[s] for s in np.ma.clump_unmasked(np.ma.masked_invalid(a))]

def fit_random_walk_model_no_boundaries(X, dt, lower_cutoff=5, upper_cutoff=245, min_len=5):
    if len(X.shape) > 1:
        pos = np.copy(X[:,0])
    else:
        pos = np.copy(X)
    pos[pos<lower_cutoff] = nan
    pos[pos>upper_cutoff] = nan
    split_pos = [x for x in using_clump(pos) if len(x) > min_len]
    best_params = map(lambda x: fit_random_walk_model(x, dt), split_pos)
    best_params = np.array(list(best_params))
    split_lens = [len(x) for x in split_pos]
    q_hat = np.average(best_params[:,0], weights=split_lens)
    tau_hat = np.average(best_params[:,1], weights=split_lens)
    return q_hat, tau_hat

def get_random_walk_likelihood(X, dt, q, tau, use_velocity=True):
    sd = q * np.sqrt(dt)
    l = 0
    if len(X.shape) > 1:
        pos = X[:,0]
    else:
        pos = X
    if use_velocity:
        v = X[:,1]
    else:
        v = np.diff(pos) / dt
    summand = np.diff(v) + (dt * (1/tau) * v[:-1])
    summand = summand ** 2
    l += np.sum(summand)
    l /= - (2 * (sd ** 2))
    log_term = (1 - X.shape[0]) * np.log(sd)
    l += log_term
    return l

def fit_random_walk_model_empirically(X, dt, samples=100, tau_range=(0, 10), q_range=(0, 100), use_velocity=True):
    taus = np.sort(np.random.uniform(tau_range[0], tau_range[1], samples))
    qs = np.sort(np.random.uniform(q_range[0], q_range[1], samples))
    likelihoods = {}
    v = np.diff(X[:,0]) / dt
    if use_velocity:
        v = X[:,1]
    base_diffs = np.diff(v)
    
    best_tau = 0
    best_q = 0
    max_likelihood = float("-inf")
    for tau, q in product(taus, qs):
        likelihood = get_random_walk_likelihood(X, dt, q, tau, use_velocity=use_velocity)
        likelihoods[tau, q] = likelihood
        if likelihood > max_likelihood:
            max_likelihood = likelihood
            best_tau = tau
            best_q = q
    likelihoods = pd.Series(likelihoods).reset_index()
    likelihoods.columns = ["tau","q","likelihood"]
    return best_q, best_tau, likelihoods

def fit_q_empirically(X, dt, tau, samples=1000, q_range=(0, 200), use_velocity=True):
    qs = np.sort(np.random.uniform(q_range[0], q_range[1], samples))
    likelihoods = []
    v = np.diff(X[:,0]) / dt
    if use_velocity:
        v = X[:,1]
    base_diffs = np.diff(v)
    
    best_q = 0
    max_likelihood = float("-inf")
    for q in qs:
        # p(X|sigma)           
        likelihood = get_random_walk_likelihood(X, dt, q, tau, use_velocity=use_velocity)
        likelihoods.append(likelihood)
        if likelihood > max_likelihood:
            max_likelihood = likelihood
            best_q = q
            
    likelihoods = pd.Series(likelihoods, index=qs).reset_index()
    likelihoods.columns = ["q","likelihood"]
    return best_q, likelihoods

def fit_tau_empirically(X, dt, q, samples=1000, tau_range=(0, 20), use_velocity=True):
    taus = np.sort(np.random.uniform(tau_range[0], tau_range[1], samples))
    likelihoods = []
    v = np.diff(X[:,0]) / dt
    if use_velocity:
        v = X[:,1]
    base_diffs = np.diff(v)
    
    best_tau = 0
    max_likelihood = float("-inf")
    for tau in taus:
        likelihood = get_random_walk_likelihood(X, dt, q, tau, use_velocity=use_velocity)
        likelihoods.append(likelihood)
        if likelihood > max_likelihood:
            max_likelihood = likelihood
            best_tau = tau
            
    likelihoods = pd.Series(likelihoods, index=taus).reset_index()
    likelihoods.columns = ["tau","likelihood"]
    return best_tau, likelihoods
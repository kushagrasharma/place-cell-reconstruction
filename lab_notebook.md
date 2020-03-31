# Lab Notebook

## Research problem
Our research problem is to investigate decoding of hippocampal place cell data to location and velocity. We want to examine current methods such as maximum likelihood estimators, particle filters, and numerical stochastic differential equation solvers. We will compare the accuracy of each of these methods, and propose novel methods for decoding location and velocity that have greater accuracy and/or computational efficiency for on-line decoding. 
A mathematical formulation - we seek computational methods for solving the filtering problem:
$$p(x_t|y_{1:t})$$ (1)
Where $$y_i$$ is the neural spike train at time $$i$$ and $$x_i$$ is the real-world quantity at time $$i$$ encoded in the neural spike train, in our case, the location of a rat.
## Data
We are using the [CRCNS hc-3](https://crcns.org/data-sets/hc/hc-3/about-hc-3) dataset, which contains hippocampal recordings from Long-Evans rats as the rats run along a linear track of $$250$$cm. Spike sorting has already been done. Hippocampal recordings are done at $$20$$kHz, and location is provided by using computer vision to detect the position of LED's placed on the rat. Location data is sampled at a rate of $$39$$Hz.
### Data Processing
We bucket our data into time and location bins in order to standardize and discretize our data for analysis. We use standard hyperparameter tuning methods (train-holdout-test sets, tune on holdout set) for choosing our time and location bins for particular models. 
## General Modelling Assumptions
### Spike Train Modelling
We model our spike trains as Poisson point processes, where the number of spikes for neuron $$i$$ at time $$t,y_t^{(i)}\sim \text{Pois} \big(f_i(x_t) \Delta t\big)$$. We also assume each neuron's spikes at each time $t$ are i.i.d.
Let $$T=\{\tau | x_\tau = x_t\}$$. Then the mean firing rate for neuron $$i$$ at location $$x_t$$ is $$f_i(x_t)=\frac{\sum_{\tau\in T}y_\tau^(i)}{|T|\Delta t}$$. The computation of the mean firing rates is generally done before the model is run on a training set, and fed as a parameter to the model.
### Place Field Smoothing Methods
* Gaussian kernel smoothing
    * Kernel smoothing is a method of estimating a real valued function $$f:\mathbb{R}^n\rightarrow \mathbb{R}$$ from data
    * The method uses a kernel function (window function) computed on neighboring data points $$x_i$$ to $$x$$ such that you get a weighted average for your result $$f(x)$$
    * This allows us to estimate the real valued function $$f$$ as $$\hat{f}(x)=\frac{1}{n}\sum_{i=1}^n K(x,x_i)$$ where $$K(x,x_i)$$ is your kernel function and $$\{x_1,...,x_n\}$$ is your dataset
    * One type of kernel is the Gaussian kernel, which uses $$K(x,x_i)=exp(-\frac{(x-x_i)^2}{2\sigma^2})$$
    * Here, we would be smoothing firing rate modulated by the distance from the location we're at
    * This has the advantage of not assuming that our data is drawn from a particular parametric distrubtion; namely the mixture model, so it could account for unexpected data in a better way
* Fitting a Gaussian mixture model
    * Each place cell contains $$1+$$ place fields, each of which is a mixture of Gaussians, where each Gaussian has an amplitude which represents the maximum firing rate at the center of the place field
    * The place field for neuron $$i$$ represents the mean firing rate $$f_i(x)$$ for location $$x$$
    * We use expectation-maximixation to fit our seen data to a model with $$k$$ place fields, each of which is centered at $$\mu_i$$, has variance $$\sigma_i$$, and has amplitude $$\alpha_i$$
    * We fit each of these parameters, with a maximum value for the number of place fields estimated empirically, using EM

## Maximum likelihood estimator methods

### General method
We assume the Markov property holds, with our latent state being location and our obervable variable is the spike train at each time step. We want to estimate the distribution $$p(x_t|y_t)$$.
$$p(x_t|y_t)=\frac{p(y_t|x_t)p(x_t)}{p(y_t)}$$
What we really want is the most likely location:
$$\hat{x}=\text{argmax}_x p(x|y_t)$$
$$\iff \hat{x}=\text{argmax}_x \log p(x|y_t)$$
We assume a uniform prior $$p(x)$$, and since $$p(y_t)$$ is independent of $$x$$, we can simplify:
$$\hat{x}=\text{argmax}_x \log(p(y_t|x))$$
$$=\text{argmax}_x \sum_{i=1}^N \log(p(y_t^{(i)}|x))$$ for $$N$$ neurons, since our individual neurons are i.i.d.
$$=\text{argmax}_x \sum_{i=1}^N \log\big(\frac{(f_i(x)\Delta t)^{y_t^{(i)}} e^{-f_i(x)\Delta t}}{y_t^{(i)}!}\big)$$
$$=\text{argmax}_x \sum_{i=1}^N (y_t^{(i)}\log(f_i(x)\Delta t) - f_i(x)\Delta t)$$
To compute $$\hat{x}$$, since we have a finite number of locations, we can simply calculate the above quantity for all $$x$$.
### Continuity Contraint Method
One shortcoming of the previous method is that it fails to take into account any previous data or predicted locations. One solution is incorporating our prediction from the previous timestep, and implementing a continuity contraint such that the predicted location from the current timestep is probabalistically bound to be close to the previous prediction. We do this by now calculating 
$$\hat{x}=\text{argmax}_x p(x_t|y_t,x_{t-1})$$
$$=\text{argmax}_x p(x_t|y_t)p(x_{t-1}|x_t)$$
Where $$p(x_{t-1}|x_t)=C\exp \big(\frac{-\left\lVert x_{t-1}-x_t \right\rVert}{2\sigma_{x_{t-1}}^2}\big )$$ and $$\sigma_{x_{t-1}}=K\big (\frac{v_{x_{t-1}}}V\big)^{\frac12}$$ where $$K,V$$ are hyperparameters, $$C$$ is a normalizing constant, and $$v_{x_{t-1}}$$ is the average speed at location $$x_{t-1}$$.
### Shortcomings of These Approaches
* We fail to take into account past spike trains $$y_{1:t-2}$$ because of our Markovian assumption
* Computation time is quite high for large ranges or small length intervals, since we need to compute the posterior probability for all possible locations
## Particle Filtering Methods
### Importance Sampling
### Sequential Importance Sampling
### Basic Particle Filter
### Bootstrap Particle Filter
## Rat motion model
### Simple Diffusion Model
$$dx =  \sigma_x \times dW_t$$
$$dW_t \sim \mathcal{N} (0, dt)$$
$$dt$$ somewhere around $$0.1-0.001$$, test and check.
$$p(x_t | x_{t-dt} ) = \mathcal{N} ( x_t; x_{t-1}, \sigma_x^2 \times dt )$$
### Integrated Random Walk Model
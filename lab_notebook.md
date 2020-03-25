# Lab Notebook

## Research problem
Our research problem is to investigate decoding of hippocampal place cell data to location and velocity. We want to examine current methods such as maximum likelihood estimators, particle filters, and numerical stochastic differential equation solvers. We will compare the accuracy of each of these methods, and propose novel methods for decoding location and velocity that have greater accuracy and/or computational efficiency for on-line decoding. 
A mathematical formulation - we seek computational methods for solving the filtering problem:
$$p(x_t|y_{1:t})$$ (1)
Where $$y_i$$ is the neural spike train at time $$i$$ and $$x_i$$ is the real-world quantity at time $$i$$ encoded in the neural spike train, in our case, the location of a rat.
## Data
We are using the [CRCNS hc-3](https://crcns.org/data-sets/hc/hc-3/about-hc-3) dataset, which contains hippocampal recordings from Long-Evans rats as the rats run along a linear track of $$250$$cm. Spike sorting has already been done. Hippocampal recordings are done at $$20$$kHz, and location is provided by using computer vision to detect the position of LED's placed on the rat. Location data is sampled at a rate of $$39$$Hz.
### Data Processing
We bucket our data into time and location bins in order to standardize and discretize our data for analysis. We use standard hyperparameter tuning methods for choosing our time and location bins for particular models. 
## Maximum likelihood estimator methods

### General method

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


import numpy as np

import torch
import torch.distributions as dist
import pyro
import pyro.distributions as pyro_dist
from pyro.nn import PyroSample, PyroModule

# for VI posterior sampling and hmc
from pyro.infer import Predictive, MCMC, NUTS 

class BayesianRegression(PyroModule):
    """Bayesian linear regression model"""
    def __init__(
        self, 
        in_features, 
        out_features=1,
        bias=True,
        bias_prior=None,
        weight_prior=None,
        sigma_prior=None,
    ):
        
        super().__init__()
        self._bias_flag = bias
        # setup the linear model
        self.linear = PyroModule[torch.nn.Linear](in_features, out_features, bias=bias)
        # setup priors
        self._weight_prior = weight_prior or pyro_dist.Normal(0., 1.).expand([out_features, in_features])#.to_event(2)
        self.linear.weight = PyroSample(self._weight_prior)
        if self._bias_flag:
            self._bias_prior = bias_prior or pyro_dist.Normal(0., 2.).expand([out_features])#.to_event(1)
            self.linear.bias =  PyroSample(self._bias_prior)
        self._sigma_prior = sigma_prior or pyro_dist.HalfCauchy(scale=torch.tensor([1.0]))
            
    
    def forward(self, x, y=None):
        sigma = pyro.sample("sigma",  self._sigma_prior)
        mean = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", pyro_dist.Normal(mean, sigma), obs=y)
        return mean
    
    
    def generate_synthetic_data(x_data, num_synthetic_samples=100):
        """returns: generated parameters and synthethic samples"""
        w0 = self._weight_prior.sample()
        sigma0 = self._sigma_prior.sample()
        if self._bias_flag:
            b0 = self._bias_prior.sample()
            params = (b0, w0, sigma0)
            mean = b0 + torch.mm(x_data, w0.T)
        else:
            params = (w0, sigma0)
            mean = torch.mm(x_data, w0.T)
        
        return params, dist.Normal(mean, sigma).sample(torch.Size(num_synthetic_samples))
        

def run_vi(x_data, y_data, vi_engine, model, guide, num_iterations=int(1e4), num_post_samples=int(1e3)): 
    """
    Runs VI for a given number of iterations -- num_iterations
    Params:
        vi_engine -- anything that can do the optimisation and has .step method
        model -- generative model to do inference on
        guide -- variational family
    Returns:
        ELBOs (average) and Samples (if num_post_samples > 0)
    """
    pyro.clear_param_store()
    elbos = []
    num_samples = x_data.shape[0]
    for j in range(num_iterations):
        # calculate the loss and take a gradient step
        loss = vi_engine.step(x_data, y_data) / num_samples
        elbos.append(-loss)
    
    guide.requires_grad_(False)
    if num_post_samples == 0 or num_post_samples is None:
        return elbos
    else:
        predictive = Predictive(
            model, guide=guide, num_samples=num_post_samples, return_sites=("linear.weight", "linear.bias", "sigma",)
        )
        samples = predictive(x_data)
        return elbos, samples
    
    
def run_hmc(x_data, y_data, model, num_samples=1000, warmup_steps=200):
    """
    Runs NUTS
    returns: samples
    """
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc.run(x_data, y_data)
    hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
    hmc_samples["linear.weight"] = hmc_samples["linear.weight"].reshape(num_samples, -1)
    return hmc_samples
    
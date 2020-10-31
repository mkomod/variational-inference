import numpy as np

import torch
import torch.distributions as dist
import pyro
import pyro.distributions as pyro_dist
from pyro.nn import PyroSample, PyroModule

# for VI posterior sampling and hmc
from pyro.infer import Predictive, MCMC, NUTS 

__all__ = ["BayesianRegression", "run_vi", "run_hmc"]


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
        self.bias_flag = bias
        # setup the linear model
        self.linear = PyroModule[torch.nn.Linear](in_features, out_features, bias=bias)
        # setup priors
        self._weight_prior = weight_prior or pyro_dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2)
        self.linear.weight = PyroSample(self._weight_prior)
        if self.bias_flag:
            self._bias_prior = bias_prior or pyro_dist.Normal(0., 2.).expand([out_features]).to_event(1)
            self.linear.bias =  PyroSample(self._bias_prior)
        self._sigma_prior = sigma_prior or pyro_dist.HalfCauchy(scale=torch.tensor([1.0]))
            
    
    def forward(self, x, y=None):
        sigma = pyro.sample("sigma",  self._sigma_prior)
        mean = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", pyro_dist.Normal(mean, sigma), obs=y)
        return mean
    
    
    def generate_synthetic_data(self, x_data):
        """returns: generated parameters, synthethic samples and """
        trace = pyro.poutine.trace(self).get_trace(x=x_data)
        y_synthetic = trace.nodes["obs"]["value"]
        if self.bias_flag:
            nodes = ["sigma", "linear.weight", "linear.bias"]
        else:
            nodes = ["sigma", "linear.weight"]
        params = torch.cat([trace.nodes[x]["value"].reshape(-1,) for x in nodes])
        return params, y_synthetic
    
    
    
        

def run_vi(
    x_data, 
    y_data, 
    vi_engine, 
    model, 
    guide, 
    num_iterations=int(1e4), 
    num_post_samples=int(1e3),
    return_elbo=True,
): 
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
        samples = torch.stack([guide.sample_latent() for i in range(num_post_samples)])
        # we dont need posterior predictives to sample from posterior!!
#         predictive = Predictive(
#             model, 
#             guide=guide, 
#             num_samples=num_post_samples, 
#             return_sites=("linear.weight", "linear.bias", "sigma",)
#         )
        if return_elbo:
            return elbos, samples
        else:
            return samples
    
    
def run_hmc(x_data, y_data, model, num_samples=1000, warmup_steps=200,):
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
    
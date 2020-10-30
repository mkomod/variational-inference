import numpy as np

import torch
import torch.distributions as dist


def diagnose_vi_VSBC(x_data, model, guide, inference_engine, replications=150):
    """ 
    params:
        model: needs to have `generate_synthetic_data` method
        inference_engine: should return samples; should accept x_data, y_synthetic
                          (partial in everything else)
    Need to figure out how to use guides to evaluate that distribution;
    currently its .cdf returns "not implemented error" any way..
                          
    returns:
        
    """
    params0, y_synthetic = model.generate_synthetic_data(x_data)
    # TODO: check if we need to take care of bias term
    post_samples = inference_engine(x_data, y_synthetic)
    post_samples_mean = torch.Tensor(post_samples.mean(axis=1))
    
    # output: p-values
    probs = torch.zeros(replications, post_samples.shape[1])
    
    if guide.__repr__() == "AutoDiagonalNormal()":
        post_samples_sd = torch.Tensor(post_samples.std(axis=1))
        norm_dist = dist.Normal(loc=post_samples_mean, scale=post_samples_sd)
            
    elif guide.__repr__() == "AutoMultivariateNormal()":
        post_samples_cov = np.cov(posterior_samples, rowvar=False)
        norm_dist = dist.MultivariateNormal(
            loc=posterior_samples_mean, covariance_matrix=post_samples_cov
        )
    else:
        raise ValueError("Guide can only be AutoDiagonalNormal or AutoMultivariateNormal")
        
    for i in range(replications):
        probs[i] = norm_dist.cdf(params0)
    return probs.numpy()


def apply_PSIS():
    pass


def check_Huggins_stuff():
    pass



import numpy as np

import torch
import torch.distributions as dist

__all__ = ["diagnose_vi_VSBC", "apply_PSIS", ]


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
    # output: p-values
    probs = torch.zeros(replications, guide.latent_dim)
        
    if guide.__repr__() == "AutoDiagonalNormal()":
        cov_fn= lambda x: x.std(axis=0)
        norm_dist_fn = dist.Normal# (loc=post_samples_mean, scale=post_samples_sd)
            
    elif guide.__repr__() == "AutoMultivariateNormal()":
        cov_fn = lambda x: np.cov(x, rowvar=False)
        norm_dist_fn = dist.MultivariateNormal
    else:
        raise ValueError("Guide can only be AutoDiagonalNormal or AutoMultivariateNormal")

    for i in range(replications):
        params0, y_synthetic = model.generate_synthetic_data(x_data)
        post_samples = inference_engine(x_data, y_synthetic)
        post_samples_mean = post_samples.mean(axis=0)
        post_samples_scale = cov_fn(post_samples)
        norm_dist = norm_dist_fn(post_samples_mean, post_samples_scale)
        probs[i] = norm_dist.cdf(params0)
        if i % 10 == 0:
            print(".", end="")
    
    return probs.numpy()


def apply_PSIS():
    # is there a method like model.log_prob? -- technically we can assume just normal and use current priors 
    # but would like it to be more generic&flexible and applicable to different


def check_Huggins_stuff():
    pass

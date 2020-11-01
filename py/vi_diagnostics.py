import numpy as np

import torch
import torch.distributions as dist
import pyro
import psis
from bayesreg_pyro import sample_and_calculate_log_impweight
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal


__all__ = ["diagnose_vi_VSBC", "apply_PSIS", ]



def SNIS_approximate(samples, log_weights, fn=lambda x: x):
    """fn - function (of params) whose approximation is to be calculated"""
    # weights add up to 1
    weights = torch.exp(log_weights).reshape(1, -1)
    return torch.mm(weights, fn(samples)).reshape(-1)


def diagnose_vi_VSBC(
    x_data, 
    model, 
    guide_class, 
    vi_num_iterations=3000,
    samples_generator=sample_and_calculate_log_impweight, 
    adjust_VI_method=None,
    replications=150, 
):
    """ 
    params:
        model: needs to have `generate_synthetic_data` method
        inference_engine: should run VI (accepts x_data and y_syn -- partial in everything else)
    samples_generator: ...
    Need to figure out how to use guides to evaluate that distribution;
    currently its .cdf returns "not implemented error" any way..
    adjust_VI_method: one of "PSIS", "SNIS" or None (for no adjustment, default)
                          
    returns: numpy array of p-values
        
    """
    # output: p-values
    
    if guide_class not in [AutoDiagonalNormal, AutoMultivariateNormal]:
        raise ValueError(
            f"Guide={guide_class} is not supported.",
            "Guide can only be AutoDiagonalNormal or AutoMultivariateNormal"
        )
    optimser = pyro.optim.Adam({"lr": 0.009})
    probs_unscaled = [] #torch.zeros(replications, guide.latent_dim)
    probs_snis_mean_only = [] # mean correction only
    probs_psis_mean_only = [] # mean correction only
    probs_snis_mean_and_std = [] # mean and std correction 
    probs_psis_mean_and_std = [] # mean and std correction 
    kappas = torch.empty(replications)
    
    for i in range(replications):
        if i % 10 == 0:
            print(".--", end="")
            
        pyro.clear_param_store()
        guide = guide_class(model)
        # synthetic data from prior
        params0, y_synthetic = model.generate_synthetic_data(x_data)
        
        # run VI on synthetic data
        _ = model.run_vi(
            x_data, y_synthetic, SVI(model, guide, optimser, loss=Trace_ELBO()), num_iterations=vi_num_iterations,
        )
        
        guide.requires_grad_(False)
        # generate samples and weights
        post_samples, log_imp_weights = samples_generator(x_data, y_synthetic, model, guide)
        #print(f"post_samples[0]={post_samples[0]}")
        psis_log_imp_weights, kappa = psis.psislw(log_imp_weights)
        psis_log_imp_weights = torch.tensor(psis_log_imp_weights)
        kappas[i] = kappa # store kappas 
        
        # unscaled p-values
        q_mean = guide.get_posterior().mean
        q_std = guide.get_posterior().stddev
        probs_unscaled.append(1 - dist.Normal(q_mean, q_std).cdf(params0))
        
        # SNIS Scaled p-values
        q_mean_SNIS = SNIS_approximate(post_samples, log_imp_weights)
        probs_snis_mean_only.append(1 - dist.Normal(q_mean_SNIS, q_std).cdf(params0))
        
        second_moment = SNIS_approximate(post_samples, log_imp_weights, lambda x: x**2)
        q_std_SNIS = torch.sqrt(second_moment - q_mean_SNIS**2)
        probs_snis_mean_and_std.append(1 - dist.Normal(q_mean_SNIS, q_std_SNIS).cdf(params0))
        
        # PSIS scaled p-values 
        q_mean_PSIS = SNIS_approximate(post_samples, psis_log_imp_weights)
        probs_psis_mean_only.append(1 - dist.Normal(q_mean_PSIS, q_std).cdf(params0))
        
        second_moment = SNIS_approximate(post_samples, psis_log_imp_weights, lambda x: x**2)
        q_std_PSIS = torch.sqrt(second_moment - q_mean_PSIS**2)
        probs_psis_mean_and_std.append(1 - dist.Normal(q_mean_PSIS, q_std_PSIS).cdf(params0))
                    
    pyro.clear_param_store()
    return {
        "probs_unscaled": torch.stack(probs_unscaled).numpy(),
        "probs_snis_mean_only": torch.stack(probs_snis_mean_only).numpy(),
        "probs_snis_mean_and_std": torch.stack(probs_snis_mean_and_std).numpy(),
        "probs_psis_mean_only": torch.stack(probs_psis_mean_only).numpy(),
        "probs_psis_mean_and_std": torch.stack(probs_psis_mean_and_std).numpy(),
        "kappas": kappas,
    }


def check_Huggins_stuff():
    pass

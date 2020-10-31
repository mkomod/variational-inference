import numpy as np

import torch
import torch.distributions as dist
import pyro
from pyro import poutine

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
        
    if guide.__repr__() not in ["AutoDiagonalNormal()", "AutoMultivariateNormal()"]:
        raise ValueError(
            f"Guide={guide.__repr__()} is not supported.",
            "Guide can only be AutoDiagonalNormal or AutoMultivariateNormal"
        )

    for i in range(replications):
        if i % 10 == 0:
            print(".", end="")
        params0, y_synthetic = model.generate_synthetic_data(x_data)
        post_samples = inference_engine(x_data, y_synthetic)
        post_samples_mean = post_samples.mean(axis=0)
        post_samples_scale = post_samples.std()
        norm_dist = dist.Normal(post_samples_mean, post_samples_scale)
        probs[i] = norm_dist.cdf(params0)
        
    return probs.numpy()


def sample_and_calculate_log_impweight(x_data, y_data, model, guide, replications=150):
    """  
    returns: log importance weights
    """
    log_impweights = torch.zeros(torch.Size([replications]))
    for i in range(replications):
        trace_guide = poutine.trace(guide).get_trace()
        # dict
        param_sample = trace_guide.nodes["_RETURN"]["value"]
        # need to evaluate the log probs so that it appears in ["_AutoDiagonalNormal_latent"]["log_prob_sum"]
        trace_guide.log_prob_sum()
        # check log prob sum!! ["_AutoDiagonalNormal_latent"]["log_prob_sum"] seems to be correct,
        # while the above <trace_guide.log_prob_sum()> equals 
        param_sample_logprob = trace_guide.log_prob_sum() #trace_guide.nodes["_AutoDiagonalNormal_latent"]["log_prob_sum"]

        cond_model = poutine.condition(model, data={"obs": y_data, **param_sample})
        trace_cond_model = poutine.trace(cond_model).get_trace(x=x_data)
        joint_logprob = trace_cond_model.log_prob_sum()
        #<=>estimated log-posterior
        log_impweights[i] = joint_logprob - param_sample_logprob
    return log_impweights


def apply_PSIS(x_data, y_data, model, guide, inference_engine, replications=150):
    pass
    # is there a method like model.log_prob? -- technically we can assume just normal and use current priors 
    # but would like it to be more generic&flexible and applicable to different
    


def check_Huggins_stuff():
    pass

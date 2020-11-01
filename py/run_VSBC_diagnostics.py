from functools import partial
import time
import argparse
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.distributions as dist
import pyro
import pyro.distributions as pyro_dist
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal, init_to_mean, AutoNormal
from pyro.infer import SVI, Trace_ELBO

import bayesreg_pyro as br
import vi_diagnostics as vid


def main(guide_name, adjustment_method, seed, replications, vi_steps, fname_prefix,):
    data = pd.read_csv("bodyfat.csv").drop("Density", axis=1)
    data = data - data.mean()

    y_data = torch.Tensor(data["Bodyfat"].values)
    x_data = torch.Tensor(data.drop("Bodyfat", axis=1).values)
    in_features = x_data.shape[1]
    
    # step 1: setup model
    pyro.clear_param_store()
    model = br.BayesianRegression(in_features, 1)
    
    # TODO: shouldnt be done this way 
    if guide_name == "AutoDiagonalNormal":
        guide_class = AutoDiagonalNormal#(model) # Stochastic Mean field
        
    else:
        guide_class = AutoMultivariateNormal#(model)
  
    # step 3: run VSBC diagnostic
    if adjustment_method == "None":
        adjustment_method = None
    if seed > 0:
        torch.manual_seed(seed)
    tic = time.time()
    result = vid.diagnose_vi_VSBC(
        x_data, 
        model, 
        guide_class=guide_class, 
        vi_num_iterations=vi_steps,
        replications=replications, 
        adjust_VI_method=adjustment_method,
    )
    toc = time.time()
    print(f"VSBC complete. Time: {(toc-tic)/60} min.")
    fname = f"results/{fname_prefix}{guide_name}_{adjustment_method}_{seed}_{replications}_{vi_steps}.pkl"
    print("Saving data... \n")
    with open(fname, 'wb') as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VSBC diagnostic")
    parser.add_argument("--guide-name", nargs="?", default="AutoDiagonalNormal", type=str)
    parser.add_argument("--adjustment-method", nargs="?", default="None", type=str)
    parser.add_argument("--seed", nargs="?", default=-1, type=int)
    parser.add_argument("--replications", nargs="?", default=500, type=int)
    parser.add_argument("--vi-steps", nargs="?", default=int(5e3), type=int)
    parser.add_argument("--fname-prefix", nargs="?", default="", type=str)
    
    args = parser.parse_args()
    main(
        args.guide_name, 
        args.adjustment_method, 
        args.seed, 
        args.replications, 
        args.vi_steps, 
        args.fname_prefix
    )



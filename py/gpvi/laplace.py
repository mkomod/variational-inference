import torch
from torch.distributions import constraints
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.contrib.gp.likelihoods.likelihood import Likelihood


class Laplace(Likelihood):
    def __init__(self, scale=None):
        super().__init__()
        scale = torch.tensor(1.) if scale is None else scale
        self.scale = Parameter(scale)
        self.set_constraint("scale", constraints.positive)


    def forward(self, f_loc, f_var, y=None):
        y_var = f_var + self.scale

        y_dist = dist.Laplace(f_loc, y_var)
        if y is not None:
            y_dist = y_dist.expand_by(y.shape[:-f_loc.dim()]).to_event(y.dim())
        return pyro.sample("y", y_dist, obs=y)

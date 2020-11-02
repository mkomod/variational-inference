import matplotlib.pyplot as plt
import torch

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist


def plot_pred(X, y, model, n_test=500):

    plt.figure(figsize=(12, 6))
    
    # plot the observed data
    plt.plot(X.numpy(), y.numpy(), "kx")

    Xtest = torch.linspace(-0.5, 5.5, n_test)
    
    # compute predictive mean and variance
    with torch.no_grad():
        if type(model) == gp.models.VariationalSparseGP:
            mean, cov = model(Xtest, full_cov=True)
        else:
            mean, cov = model(Xtest, full_cov=True, noiseless=False)

    # plot the mean function and confidence bounds
    sd = cov.diag().sqrt()
    plt.plot(Xtest.numpy(), mean.numpy(), 'r', lw=2)
    plt.fill_between(Xtest.numpy(),
                     (mean - 2.0 * sd).numpy(),
                     (mean + 2.0 * sd).numpy(),
                     color='C0', alpha=0.3)

    plt.xlim(-0.5, 5.5)

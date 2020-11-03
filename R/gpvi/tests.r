# Tests for GPVI 
# 
# Test 1: Optimisation when hyperparmeters are known and fixed
# Checks if for some given dataset (Y, X), we are able to optimise the
# variational factors, returns a plot of the result, and prints diagnostic
# information during the test. The test is passed / failed by visual inspection
#	
# Test 2: Optimisation with unknown hyperparmeters
# Check if we are able to optimise the variational factors alongside some 
# unknown hyperparmeters. Produces plots during the optimisaiton alongside
# the estiamted variational factors and hyperparmeters.
#
library(mvtnorm)
library(Rcpp)  # (optional) provides numerically stable matrix inversion

source("utils.r")
source("kernels.r")
source("gpvi.r")

d <- read.csv("data.csv")
theta <- c(2, 3, 1.5)                  
X <- d$X
Y <- d$Y
V <- function(Y, x, eta) {
    - log(eta / 2) + eta * abs(Y - x)
}


# Optimise for fixed hyperparmas ----------------------------------------------
local({
    lnu.init <- runif(2 * length(Y))
    model <- opt_elbo(X, Y, kern.gauss, V, theta, lnu.init)
    K <- kern.gauss(X, X, theta[1], theta[2])
    gpvi.plot(model$pars, X, Y, K)
})


# Optimise elbo and hyperparamas ----------------------------------------------
local({
    theta.init <- c(2, 2, 2)
    lnu.init <- runif(2 * length(Y))
    model <- gpvi.fit(X, Y, kern.gauss, V, theta.init, lnu.init)
})


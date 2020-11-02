library(mvtnorm)

source("utils.r")
source("kernels.r")
source("gpvi.r")


# Generate Synthetic Data -----------------------------------------------------
set.seed(12)
X <- seq(0, 10, length.out=50)
K_ <- kern.gauss(X, X, 2, 3)
f <- mvtnorm::rmvnorm(1, mean=rep(0, length(X)),  K_) 
bounds <- quantile(rlaplace(1000, eta=2), 0.025)
lower <- f + bounds[1]
upper <- f - bounds[1]
plot(X, f, ylim=c(-15, 15), type="b", pch=16, cex=0.7)
lines(X, upper, col="red")
lines(X, lower, col="red")
Y <- f  + rlaplace(length(X), eta=2)
points(X, Y, pch=17, col="green")


# Optimise for fixed hyperparmas ----------------------------------------------
local({
    V <- function(Y, x, eta) {
	- dnorm(Y, mean=x, sd=eta, log=TRUE)
    }
    lnu.init <- runif(2 * length(Y))
    theta <- c(0.5, 0.5, 0.5)

    # Fit GPVI using synthetic data
    model <- opt_elbo(X, Y, kern.gauss, V, theta, lnu.init)
    pars <- model$par
    K <- kern.gauss(X, X, theta[1], theta[2])
    gpvi.plot(pars, X, Y, K)
})


# Optimise elbo and hyperparamas ----------------------------------------------
local({
    V <- function(Y, x, eta) {
	# - dnorm(Y, mean=x, sd=eta, log=TRUE)
	- log(eta / 2) + eta * abs(Y - x)
    }
    theta.init <- c(0.5, 0.5, 0.5)
    lnu.init <- runif(2 * length(Y))

    # Fit GPVI using synthetic data
    model <- gpvi.fit(X, Y, kern.gauss, V, theta.init, lnu.init)
    print(model)
})


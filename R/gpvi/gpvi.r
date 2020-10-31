MC.SAMPLES <- 1e3                      # Monte Carlo Samples
N.ITERATIONS <- 50                     # Number of iterations

#' Computes the Gaussian Kernel K for vectors x1, x2
kern.gauss <- function(x1, x2, l=1, s=1) {
    sapply(x1, function(x) {
	s^2 * exp( - (x - x2)^2  / l^2)
    })
}

#' Returns the negative log-likilihood for Y | X, theta
#' We encode theta as a vector that we optimise
V <- function(x, Y, theta) {
    - dnorm(Y, x, theta[1], log=TRUE)
}

# GPVI -----------------------------------------------------------------------

# Generate synthetic data
# Y <-
# X <- 
n <- length(Y)

K <- kern.gauss(X, X)
l <- runif(n)
nu <- runif(n)

for (iteration in 1:N.ITERATIONS) {
    L <- diag(l)
    S <- solve(solve(K) + L)
    mu <- K %*% nu

    for (i in 1:n) { 
	x <- rnorm(MC.SAMPLES, mean=mu[i], S[i, i])
	v <- V(x, Y[i])
	l[i] <- mean((x - mu[i]) * v / S[i, i])
	nu[i] <- mean(((x - mu[i])^2 * v  - S[i, i] * v) / S[i, i])
    }
}


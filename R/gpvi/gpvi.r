MC.SAMPLES <- 1e3                      # Monte Carlo Samples
N.ITERATIONS <- 5                     # Number of iterations

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
X <- 1:10
Y <- 10 + 2 * sin(X) + 0.05 * X + rnorm(10, sd=2)
n <- length(Y)
theta <- 2

K <- kern.gauss(X, X)
l <- rep(0, n)
nu <- rep(0, n)

for (iteration in 1:N.ITERATIONS) {
    L <- diag(l)
    S <- solve(solve(K) + L)
    mu <- K %*% nu

    for (i in 1:n) { 
	x <- rnorm(MC.SAMPLES, mean=mu[i, ], S[i, i])
	v <- V(x, Y[i], theta)
	nu[i] <- - mean((x - mu[i, ]) * v / S[i, i])
	l[i] <- mean(((x - mu[i, ])^2 * v  - S[i, i] * v) / S[i, i])
    }
}


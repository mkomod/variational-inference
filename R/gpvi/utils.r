# Utilites for GPVI
#
library(Rcpp)

#' Plot GPVI
gpvi.plot <- function(lnu, X, Y, K) {
    p <- gpvi.params(lnu, K)
    mu <- p$mu
    S <- (p$S + t(p$S)) / 2

    # Sample from mv Gauss, compute means and 95% CI
    gp.res <- mvtnorm::rmvnorm(1000, mean=mu, sigma=S)
    gp.mean <- apply(gp.res, 2, mean)
    gp.conf <- apply(gp.res, 2, function(x) quantile(x, c(0.025, 0.975) ))

    plot(X, Y, ylim=c(-15, 15))
    lines(X, gp.mean)
    lines(X, gp.conf[1, ], col="red") 
    lines(X, gp.conf[2, ], col="red") 
}


#' Normalises parameters
norm.params <- function(p) {
    return(p / norm(p, type="F"))
}


#' Returns parameters used in GPVI
gpvi.params <- function(lnu, K) {
    n <- length(lnu) / 2
    l <- lnu[1:n]
    nu <- lnu[(n+1):(2*n)]
    L <- diag(l)
    S <- solve(solve(K) + L)
    mu <- K %*% nu
    return(list(S=S, mu=mu, l=l, nu=nu, n=n))
}


rlaplace <- function(n, m=0, eta = 1) {
    unif_vals <- runif(n)
    vals <- numeric(n)
    b = 1/eta
    for (i in 1:n) {
	vals[i] = laplaceInvCdf(unif_vals[i], m, b)
    }
    return(vals)
}


laplaceInvCdf <- function(p, m, b) {
    return(m - b * sign(p - 0.5) * log(1 - 2 * abs(p - 0.5)))
}


logDet <- function(X) {
    log(det(X))
}

# Turn to C++ for operations on matrices
Rcpp::cppFunction(depends = "RcppArmadillo", '
    arma::mat solve(arma::mat X) {
	return(arma::inv(X));
    }'
)

Rcpp::cppFunction(depends = "RcppArmadillo", '
    double logDet(arma::mat X) {
	return log(arma::det(X));
    }'
)



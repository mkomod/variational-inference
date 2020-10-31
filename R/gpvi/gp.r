# Guassian Processes ---------

#' Fits a Gaussian Process and optimises the hyperparameters
#'
#' @param y Vector of response
#' @param X Vector of indicies
#' @param X.star Vector of indicies to predict
#' @param kernel Kernel to use
#' @param k.param kernel parameter starting values
#' @param s.n Initial standard deviation
#'
#' @return gaussian.process object, containing mean of predicitons, variance
#' 	hyperparameters used, likelihood and data used.
#'
gaussian.process.fit <- function(y, X, X.star=X, 
	kernel=kernel.gauss, k.params=c(1, 1), s.n=1) {

    I.n <- diag(length(y))

    gp.lik <- function(X, kernel, s.n) {
	K <- kernel(X, X)
	A <- K + s.n^2 * I.n
	A.inv <- solve(A)

	return(0.5 * t(y) %*% A.inv %*% y + 0.5 * 
	       log(det(A)) + 0.5 * nrow(K) * log(2*pi))
    }

    res.optim <- optim(c(s.n, k.params), function(x) {
	kern <- function(x.1, x.2) kernel(x.1, x.2, x[-1])
	return(gp.lik(X, kern, x[1]))
    })

    optim.kernel.params <- res.optim$par[-1]
    optim.s.n <- res.optim$par[1]
    optim.lik <- res.optim$value

    K <- kernel(X, X, optim.kernel.params)
    A <- (K + optim.s.n^2 * diag(length(y)))
    A.inv <- solve(A)
    k.star <- kernel(X.star, X, optim.kernel.params)

    res <- list(
	y = y,
	X = X,
	X.star = X.star,
	mean = t(k.star) %*% A.inv %*% y,
	var = kernel(X.star, X.star, optim.kernel.params) - 
	    t(k.star) %*% A.inv %*% k.star,
	lik = optim.lik,
	kernel.params = optim.kernel.params,
	s.n = optim.s.n
    )

    class(res) <- "gaussian.process"
    return(res)
}


#' Gaussian Kernel
#' 
#' @param x.1 vector of indicies
#' @param x.2 vector of indicies
#' @param k.param kernel parameters, length and scale
#' 
#' @return estiamted covariance matrix
#' 
kernel.gauss <- function(x.1, x.2, k.params=c(1, 2)) {
    return(sapply(x.1, function(i) {
	k.params[1]^2 * exp( - (i - x.2)^2 / k.params[2]^2)
    }))
}



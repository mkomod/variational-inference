MC.SAMPLES <- 3e3                      # Num of MC samples
G.MC.SAMPLES <- 3e4                    # Num samples when computing gradient


#' Intermediate function for computing the gradient of l and nu
lnu_bar <- function(lnu, K, V, eta) {
    p <- gpvi.params(lnu, K)
    S <- p$S
    mu <- p$mu
    n <- p$n
    l <- p$l
    nu <- p$nu

    nu.bar <- numeric(n)
    l.bar <- numeric(n)
    for (i in 1:n) { 
	x <- rnorm(G.MC.SAMPLES, mean=mu[i, ], sd=sqrt(S[i, i]))
	v <- V(Y[i], x, eta)
	nu.bar[i] <- -mean((x - mu[i, ]) * v) / S[i, i]
	l.bar[i] <- mean(((x - mu[i, ])^2 * v - S[i, i] * v) / S[i, i])
    }

    return(list(l.bar=l.bar, nu.bar=nu.bar, S=S, mu=mu, l=l, nu=nu))
}


#' Computes the gradient wrt. the hyperparms l, nu
grad_elbo <- function(lnu, K, V, eta) {
    lnu.bar <- lnu_bar(lnu, K, V, eta)
    l.bar <- lnu.bar$l.bar
    nu.bar <- lnu.bar$nu.bar
    S <- lnu.bar$S
    l <- lnu.bar$l
    nu <- lnu.bar$nu

    # Compute the normalised gradients
    nu.grad <- norm.params(K %*% (nu - nu.bar))
    l.grad <- norm.params(0.5 * (S * S) %*% (l - l.bar))
    
    # cat("grad:", l.grad, nu.grad, "\n")
    return(c(l.grad, nu.grad))
}


#' Computes the ELBO, a quantity to be optimised
elbo <- function(lnu, K, V, eta) {
    p <- gpvi.params(lnu, K)
    S <- p$S
    mu <- p$mu
    n <- p$n

    ln.py <- numeric(n)
    for (i in 1:n) {
	x <- rnorm(MC.SAMPLES, mean=mu[i, ], sd=sqrt(S[i, i]))
	ln.py[i] <- mean(V(Y[i], x, eta))
    }

    K.inv <- solve(K)
    res <- sum(ln.py) + 
	   0.5 * (sum(diag(K.inv %*% S)) + 
	   t(mu) %*% K.inv %*% mu - log(det(S)))

    # cat("elbo:", res, "\n")
    return(res)
}


#' Optimise the ELBO for fixed hyperparams
opt_elbo <- function(X, Y, kern, V, theta, lnu.init) {
    K <- kern(X, X, theta[1], theta[2])
    opt <- optim(par=lnu.init, 
	    fn=elbo, gr=grad_elbo,
	    method="CG", 
	    control=list(fnscale=1, maxit=1e3, abstol=1e-8),
	    K=K, V=V, eta=theta[3])
    return(list(par=opt$par, elbo=opt$value, 
		X=X, Y=Y, K=K, theta=theta, lnu.init=lnu.init))
}


#' Compute the gradient of the elbo wrt. to the hyperparameters of V
grad_elbo_V <- function(eta, mu, S, Y) {
    n <- length(Y)
    d.eta <- numeric(n)
    for (i in 1:n) {
	sig <- sqrt(S[i, i])
	m <- mu[i, ]
	y <- Y[i]
	t_ <- (y - m) / sig

	d.eta[i] <- y * (2 * pnorm(t_) - 1) + 
	    m - 2 * (m * pnorm(t_) - sig * dnorm(t_))
    }
    res <- - 0.5 * (n / eta) - sum(d.eta)
    return(res)
}

#' Gradient of the ELBO wrt. to vector of hyperparams theta
grad_elbo_theta <- function(theta, lnu, X, Y, kern, V) {
    cat("params:", lnu, "\n")
    K <- kern(X, X, theta[1], theta[2])
    p <- gpvi.params(lnu, K)
    S <- p$S
    mu <- p$mu
    
    grad.e <- grad_elbo_V(theta[3], mu, S, Y)
    lnu.bar <- lnu_bar(lnu, K, V, theta[3])
    l.bar <- lnu.bar$l.bar
    nu.bar <- lnu.bar$nu.bar

    B.bar <- K + solve(diag(l.bar))
    G <- nu.bar %*% t(nu.bar) - solve(B.bar)
    dk <- kern.gauss_grad(X, X, theta[2], theta[3])
    grad.l <- -0.5 * sum(diag(G %*% dk$dl))
    grad.s <- -0.5 * sum(diag(G %*% dk$ds))

    res <- norm.params(matrix(c(grad.l, grad.s, grad.e), nrow=1, ncol=3))
    cat("grad eta:", res, "\n")
    return(res)
}


gpvi.fit <- function(X, Y, kern, V, theta.init, lnu.init) {
    lnu <- lnu.init

    elbo_theta <- function(theta, X, Y, kern, V) {
	opt.e <- opt_elbo(X, Y, kern, V, theta, lnu.init)
	lnu <<- opt.e$par
	res <- opt.e$elbo
	K <- opt.e$K
	cat("EBLO:", res, "\n")
	gpvi.plot(opt.e$par, X, Y, K)
	return(res)
    }

    grad_elbo_theta_ <- function(theta, X, Y, kern, V) {
	grad_elbo_theta(theta, lnu, X, Y, kern, V)
    }

    opt <- optim(par=theta.init,
	fn=elbo_theta, gr=grad_elbo_theta_,
	method="CG", 
	control=list(fnscale=1, maxit=1e3, abstol=1e-2),
	X=X, Y=Y, kern=kern, V=V)

    return(c(opt$par, lnu))
}


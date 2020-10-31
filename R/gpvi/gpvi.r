library(mvtnorm)

MC.SAMPLES <- 3e3                      # Monte Carlo Samples
G.MC.SAMPLES <- 3e4

#' Computes the Gaussian Kernel K for vectors x1, x2
kern.gauss <- function(x1, x2, l=4, s=9) {
    sapply(x1, function(x) {
	s^2 * exp( - (x - x2)^2  / l^2)
    })
}

# GPVI -----------------------------------------------------------------------
set.seed(1234)
X <- 1:10
Y <- 10 + 2 * sin(X) + 0.05 * X + rnorm(length(X), sd=0.5)
n <- length(Y)
plot(X, Y, ylim=c(0, 15))

V <- function(Y, x, eta) {
    - dnorm(Y, mean=x, sd=eta, log=TRUE)
}

compute_lnu_bar <- function(lnu, eta) {
    nu.bar <- numeric(n); l.bar <- numeric(n)
    nu.grad <- numeric(n); l.grad <- numeric(n)
    l <- lnu[1:n]
    nu <- lnu[(n+1):(2*n)]
    L <- diag(l)
    S <- solve(solve(K) + L)
    mu <- K %*% nu
    for (i in 1:n) { 
	x <- rnorm(G.MC.SAMPLES, mean=mu[i, ], sd=sqrt(S[i, i]))
	v <- V(Y[i], x, eta)
	nu.bar[i] <- -mean((x - mu[i, ]) * v ) / S[i, i]
	l.bar[i] <- mean(((x - mu[i, ])^2 * v  - S[i, i] * v) / S[i, i])
    }
    return(c(l.bar, nu.bar))
}

compute_grad <- function(lnu, eta) {
    lnu.bar <- compute_lnu_bar(lnu, eta)
    l.bar <- lnu.bar[1:n]
    nu.bar <- lnu.bar[(n+1):(2*n)]
    l <- lnu[1:n]
    nu <- lnu[(n+1):(2*n)]
    L <- diag(l)
    S <- solve(solve(K) + L)

    nu.grad <- K %*% (nu - nu.bar)
    l.grad <- 0.5 * (S * S) %*% (l - l.bar)

    nu.grad <- nu.grad / norm(nu.grad, type="F")
    l.grad <- l.grad / norm(l.grad, type="F")

    return(c(l.grad, nu.grad))
}


F.q <- function(lnu, eta) {
    l <- lnu[1:n]
    nu <- lnu[(n+1):(2*n)]
    L <- diag(l)
    S <- solve(solve(K) + L)
    mu <- K %*% nu
    ln.py <- numeric(n)
    for (i in 1:n) {
	x <- rnorm(MC.SAMPLES, mean=mu[i, ], sd=sqrt(S[i, i]))
	ln.py[i] <- mean(V(Y[i], x, eta))           # ln p(y_n | x_n, theta)
    }
    K.inv <- solve(K)
    res <- sum(ln.py) + 0.5 * ( sum(diag(K.inv %*% S)) + t(mu) %*% K.inv %*% mu
			- log(det(S)) )
    return(res)
}

# Optimise Hyperparms ----------------------------------------------------------


#' v is the variance
grad.eta <- function(eta, mu, S, y) {
    d.eta <- numeric(n)
    for (i in 1:n) {
	sig <- sqrt(S[i, i])
	mu. <- mu[i, ]
	y. <- y[i]
	t. <- (y. - mu.) / sig
	d.eta[i] <- y. * (2* pnorm(t.) - 1) + mu.  -2*(mu.*pnorm(t.) -sig*dnorm(t.))
    }
    return(- 0.5 * (n / eta) - sum(d.eta))
}

grad.kern.l <- function(x1, x2, l=4, s=25) {
    res <- sapply(x1, function(x) {
	s^2 * 2 * (x - x2)^2  / l^3 * exp( - (x - x2)^2  / l^2)
    })
    return(res)
}

grad.kern.s <- function(x1, x2, l=4, s=25) {
    res <- sapply(x1, function(x) {
	2 * s * exp( - (x - x2)^2  / l^2)
    })
    return(res)
}

lnu.star <- c()
set.seed(1)
l <- runif(n)                         # l.0
nu <- runif(n)                        # nu.0

obj.theta <- function(theta) {
    print(theta)
    K <<- kern.gauss(X, X, theta[1], theta[2])
    opt.pars <- optim(par=c(l, nu), 
	fn=F.q, gr=compute_grad,
	method="CG", 
	control=list(fnscale=1, maxit=1e3, abstol=1e-8),
	eta=theta[3]
    )
    lnu.star <<- opt.pars$par
    l <<- lnu.star[1:n]
    nu <<- lnu.star[(n+1):(2*n)]
    return(opt.pars$value)
}

grad.theta <- function(theta) {
    kern.s <- theta[1]
    kern.l <- theta[2]
    eta <- theta[3]
    lnu.star.bar <- compute_lnu_bar(lnu.star, eta) 
    l.star.bar <- lnu.star.bar[1:n]
    nu.star.bar <- lnu.star.bar[(n+1):(2*n)]

    L <- diag(l)
    S <- solve(solve(K) + L)
    mu <- K %*% nu

    B.bar <- K + solve(diag(l.star.bar))
    G <- nu.star.bar %*% t(nu.star.bar) - solve(B.bar)
    grad.s <- -0.5 * sum(diag(G %*% grad.kern.s(X, X, kern.l, kern.s)))
    grad.l <- -0.5 * sum(diag(G %*% grad.kern.l(X, X, kern.l, kern.s)))
    grad.e <- grad.eta(eta, mu, S, Y)
    return(c(grad.s, grad.l, grad.e))
}

theta.0 <- c(1, 1, 1)
optim(theta.0, fn=obj.theta, gr=grad.theta, method="CG", 
      control=list(fnscale=-1))



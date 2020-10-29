# -----------------------------------------------------------------------------
# Title: Variational Inference: A review for Statisticians
# Author: Blei
# Year: 2018
# -----------------------------------------------------------------------------

set.seed(1234)

sig <- 5; K <- 3; n <- 1e3
mu.k <- rnorm(K, 0, sig)
c.n <- sample(1:K, n, replace=T, prob=rep(1/K, K))
x.n <- rnorm(n, mu.k[c.n], 1)

plot(density(x.n))

its <- 50
m <- runif(K, -5, 5)
s.2 <- runif(K, 1, 5)
phi <- matrix(runif(n * K), ncol=K)
e <- numeric(its)

for (j in 1:its) {
    for (i in 1:n) {
	for (k in 1:K) {
	    phi[i, k] = exp(m[k] * x.n[i] - (s.2[k] + m[k]^2)/2)
	}
    }
    phi <- phi / apply(phi, 1, sum)    # Normalise
    t.phi.k <- apply(phi, 2, sum)
    for (k in 1:K) {
	m[k] <- ( t(phi[ , k]) %*% x.n ) / ( 1/sig^2 + t.phi.k[k] )
	s.2[k] <- 1 / (1/sig^2 + t.phi.k[k])
    }
    e[j] <- elbo(m, s.2, phi)                  # MC ELBO Estimate
    plot(e, type="l")
}

sort(m)
sort(mu.k)
t.phi.k / n

elbo <- function(m, s.2, phi) {
    n.mc <- 5e2
    elbo.hat <- numeric(n.mc)
    for (j in 1:n.mc) {
	cs <- numeric(n)
	mu.k <- numeric(K)
	for (i in 1:n) {
	    cs[i] <- sample(1:K, 1, replace=T, phi[i, ])
	}
	for (k in 1:K) {
	    mu.k[k] <- rnorm(1, m[k], sqrt(s.2[k]))
	}
	elbo.hat[j] <- log.p(mu.k, cs) + lik.x(x.n, mu.k, cs) -
	    log.q(mu.k, cs, m, s.2, phi)
    }
    mean(elbo.hat)
}

log.q <- function(mu.k, cs, m, s.2, phi) {
    sum(dnorm(mu.k, m, sqrt(s.2), log=T)) +
    sum(sapply(1:n, function(i) log(phi[i, cs[i]])))
}

log.p <- function(mu.k, cs) {
    sum(dnorm(mu.k, 0, sig, log=T)) +
    sum(n * log(1 / K))
}

lik.x <- function(x.n, mu.k, cs) {
    sum(dnorm(x.n, mu.k[cs], 1, log=T))
}


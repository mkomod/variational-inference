# -------------------------------------------------------------------------------
# Title: Variational Inference: A review for Statisticians
# Author: Blei
# Year: 2018
# -------------------------------------------------------------------------------

sig <- 5; K <- 5; n <- 1e4
mu.k <- rnorm(K, 0, sig)
c.n <- sample(1:K, n, replace=T, prob=rep(1/K, K))
x.n <- rnorm(n, mu.k[c.n], 1)

plot(density(x.n))

m <- runif(K, -5, 5)
s.2 <- runif(K, 0, 5)
phi <- matrix(runif(n * K), ncol=K)

for (j in 1:50) {
    for (i in 1:n) {
	for (k in 1:K) {
	    phi[i, K] = exp(m[k] * x.n[i] - (s.2[k] + m[k])/2)
	}
    }
    for (k in 1:K) {
	t.phi.k <- apply(phi, 2, sum)
	m[k] <- sum(phi[ , k] * x.n) / (1/ sig^2 + t.phi.k[k])
	s.2[k] <- 1 / (1/ sig^2 + t.phi.k[k])
    }
    # Missing - Compute ELBO
}


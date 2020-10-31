
# Returns a 5x5 matrix
kern.gauss(1:5, 1:5)

l.star <- pars[1:n]
nu.star <- pars[(n+1):(2*n)]
L.star <- diag(l.star)
S.star <- solve(solve(K) + L.star)
mu.star <- K %*% nu.star

S.star <- (S.star + t(S.star)) / 2

gp.res <- rmvnorm(100, mean=mu.star, sigma=S.star)
gp.mean <- apply(gp.res, 2, mean)
gp.conf <- apply(gp.res, 2, function(x) quantile(x, c(0.025, 0.975) ))

plot(X, Y, ylim=c(-1, 20))
lines(X, gp.mean)
lines(1:10, gp.conf[1, ], col="red") 
lines(1:10, gp.conf[2, ], col="red") 



# GP -------------------------------------------------------------------------

gp.fit <- gaussian.process.fit(Y, X, seq(1, 10, length.out=30))
gp.res <- rmvnorm(100, mean=gp.fit$mean, sigma=gp.fit$var)
gp.mean <- apply(gp.res, 2, mean)
gp.conf <- apply(gp.res, 2, function(x) quantile(x, c(0.025, 0.975) ))

plot(X, Y, ylim=c(-1, 20))
lines(seq(1, 10, length.out=30), gp.mean)
lines(seq(1, 10,length.out=30), gp.conf[1, ], col="red") 
lines(seq(1, 10,length.out=30), gp.conf[2, ], col="red") 





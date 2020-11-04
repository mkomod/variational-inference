library(mvtnorm)
library(Rcpp)
library(parallel)

source("utils.r")
source("kernels.r")
source("gpvi.r")

VSBC.SAMPLES <- 1e3
MC.CORES <- 4

theta <- c(2, 3, 1.5)                   # default hyperparmeters
d <- read.csv("data.csv")
X <- d$X
Y <- d$Y
f <- d$f
num.params <- length(X)
V <- function(Y, x, eta) {
    - log(eta / 2) + eta * abs(Y - x)
}
K <- kern.gauss(X, X, theta[1], theta[2])


# Plot data
pdf("./plots/gpvi_fit.pdf", width=12, height=7)
par(mar=c(4,3,0,0))
local({
    bounds <- quantile(rlaplace(1000, eta=1.5), 0.025)
    lower <- f + bounds[1]
    upper <- f - bounds[1]
    plot(X, f, type="b", 
	 pch=16, cex=0.7, bty="n",
	 ylim=c(-10, 5), ylab="")
    polygon(c(X, rev(X)), c(upper, rev(lower)), 
	    col=rgb(0, 0, 0, 0.2), border=NA)
    points(X, Y, pch=17, col="blue")
    legend(9, -6, legend=c("f(x)", "Y", "f*(x)"), col=c(1, "blue", "red"), 
	   pch=c(16, 17, NA), lty=c(1, NA, 1))
})

# Making predictions ---------------------------------------------------------
local({
    m1 <- opt_elbo(X, Y, kern.gauss, V, theta, lnu.init=runif(2 * length(Y)))
    p <- gpvi.params(m1$par, K)
    mu <- p$mu
    S <- (p$S + t(p$S)) / 2
    gp.res.p <- rmvnorm(1000, mu, S)
    gp.mean.p <- apply(gp.res.p, 2, mean)
    gp.conf.p <- apply(gp.res.p, 2, function(x) quantile(x, c(0.025, 0.975)))

    lines(X, gp.mean.p, col="red")
    polygon(c(X, rev(X)), c(gp.conf.p[2,], rev(gp.conf.p[1,])), 
    col=rgb(1, 0, 0, 0.1), border=NA)
})
dev.off()

# VSBC Diagnostic ------------------------------------------------------------
vsbc <- parallel::mclapply(1:VSBC.SAMPLES, function(j) {
    # Generate new dataset with fixed hyperparmeters
    f <- mvtnorm::rmvnorm(1, mean=rep(0, length(X)), K)
    Y <- f  + rlaplace(length(X), eta=theta[3])
    
    # Fit model, i.e. estimate theta
    m1 <- opt_elbo(X, Y, kern.gauss, V, theta, lnu.init=runif(2 * length(Y)))
    p <- gpvi.params(m1$par, K)
    mu <- p$mu
    S <- (p$S + t(p$S)) / 2
    
    # VSBC diagnostic
    res <- numeric(num.params)
    for (i in 1:num.params) {
	res[i] <- pnorm(f[i], mu[i], sqrt(S[i, i]))
    }

    cat(".")
    return(res)
}, mc.cores=MC.CORES)

vsbc.mat <- t(sapply(vsbc, function(x) x))
save(vsbc.mat, mu, S, file="results.Rdata")

vsbc.pvalues <- numeric(num.params)
for (i in 1:num.params) {
    vsbc.pvalues[i] <- ks.test(vsbc.mat[ , i], 1-vsbc.mat[ , i])$p.value
}

# pdf("./plots/gpvi_vsbc.pdf", width=12, height=3)
# par(mfrow=c(1,5), mar=c(3, 4, 4, 2))
# hist(vsbc.mat[ , 1], main=expression(f[1]), xlab="")
# par(mar=c(3, 3, 4, 2))
# hist(vsbc.mat[ , 5],  main=expression(f[5]), xlab="")
# hist(vsbc.mat[ , 10], main=expression(f[10]), xlab="")
# hist(vsbc.mat[ , 15], main=expression(f[15]), xlab="")
# hist(vsbc.mat[ , 20], main=expression(f[20]), xlab="")
# dev.off()



# Making predictions ---------------------------------------------------------
x.star <- seq(0, 10, length.out=100)
k.X.xstar <- kern.gauss(X, x.star, l=theta[1], s=theta[2])
k.xstar.X <- kern.gauss(x.star, X, l=theta[1], s=theta[2])
k.xx.star <- kern.gauss(x.star, x.star, l=theta[1], s=theta[2])

f.star <- k.X.xstar %*% solve(K + diag(diag(S))) %*% Y
V.star <- k.xx.star - t(k.xstar.X) %*%  solve(K + diag(diag(S))) %*% k.xstar.X
V.star <- V.star %*% t(V.star) / 2

gp.res <- rmvnorm(1000, f.star, V.star)
gp.mean <- apply(gp.res, 2, mean)

lines(x.star, gp.mean, col="purple")



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
local({
    bounds <- quantile(rlaplace(1000, eta=1.5), 0.025)
    lower <- f + bounds[1]
    upper <- f - bounds[1]
    plot(X, f, type="b", 
	 pch=16, cex=0.7, bty="n",
	 ylim=c(-10, 5), ylab="")
    polygon(c(X, rev(X)), c(upper, rev(lower)), 
	    col=rgb(1, 0, 0.5, 0.1), border=NA)
    points(X, Y, pch=17, col="blue")
    legend(9, -6, legend=c("f(x)", "Y"), col=c(1, "blue"), 
	   pch=c(16, 17), lty=c(1, NA))
})


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



vsbc.pvalues <- numeric(num.params)
for (i in 1:num.params) {
    vsbc.pvals[i] <- ks.test(p.2[ , i], 1-p.2[ , i])$p.value
}


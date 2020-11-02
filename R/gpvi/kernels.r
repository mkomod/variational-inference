# Kernels

kern.gauss <- function(x1, x2, l, s) {
    res <- sapply(x1, function(x) {
	s^2 * exp( - (x - x2)^2  / l^2)
    })
    return(res)
}


#' Gradient of the Kernel wrt. 
kern.gauss_grad <- function(x1, x2, l, s) {
    dl <- sapply(x1, function(x) {
	s^2 * 2 * (x - x2)^2  / l^3 * exp( - (x - x2)^2  / l^2)
    })
    ds <- sapply(x1, function(x) {
	2 * s * exp( - (x - x2)^2  / l^2)
    })
    return(list(dl=dl, ds=ds))
}


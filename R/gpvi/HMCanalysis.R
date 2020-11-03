
fit <- readRDS(here("R/gpvi/results/fit-fixedHyperData.rds"))

fit
library(shiny)
library(shinystan)
my_shinystan <- as.shinystan(fit)
launch_shinystan(my_shinystan)

sim_summary = as.data.frame(summary(fit)[[1]])
f_hmc = sim_summary$`50%`[(length(X)+2):(2*length(X)+1)]

plot(X, f, ylim=c(-7.5, 4.5), type="b", pch=16, cex=0.7)
points(X, Y, pch=17, col="green")
lines(x = X, y = f_hmc, col = "blue")
lines(x = X, y = mu, col = "red")

legend("topleft", legend = c("f(x)","y","GPVI", "HMC"), col=c("black", "green", "red", "blue"), 
       pch = c(16,17,NA,NA), lty = 1)

#mu and sigma from VI
load(file = "results.Rdata")
parameters = matrix(0,nrow = 20, ncol = 2 )
for (i in 1:20){
  parameters[i,] = c(mu[i], sqrt(S[i,i]))
}
parameters

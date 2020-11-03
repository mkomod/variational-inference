library(ggplot2)
library(StanHeaders)
library(rstan)
library(here)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# Recover toy dataset
d <- read.csv("data.csv")
X = d$X
f = d$f
Y = d$Y
e = d$e

# Plot it
bounds <- quantile(rlaplace(1000, eta=1.5), 0.025)
lower <- f + bounds[1]
upper <- f - bounds[1]
plot(X, f, ylim=c(-7.5, 4.5), type="b", pch=16, cex=0.7)
lines(X, upper, col="red")
lines(X, lower, col="red")
points(X, Y, pch=17, col="green")


#Fit the model in stan given the true parameters, that unfortunately need to be substituted by hand in stan code

pred_data = list(  N1= length(X),
                   x1 = X,
                   y1 = as.vector(Y),
                   N2 = 1,
                   x2 = as.array(20.1),
                   rho = 2,
                   alpha = 3 ,
                   eta_inv = 1.0/1.5 
  )


pred_fit = stan(file = "FixedHyperHMC.stan",data = pred_data, chains = 4, iter = 1000)

if(!dir.exists(here("R/gpvi/results"))) {
  dir.create(here("R/gpvi/results"))
}

saveRDS(
  pred_fit, 
  file = here(paste0("R/gpvi/results/", "fit-fixedHyperData", ".rds"))
)

print(pred_fit)
sim_summary = as.data.frame(summary(pred_fit)[[1]])

x_low = sim_summary$`2.5%`[22:(22+length(X)-1)]
x_med = sim_summary$`50%`[22:(22+length(X)-1)]
x_upp = sim_summary$`97.5%`[22:(22+length(X)-1)]

x_med

color = "blue"
points(x = X, y = x_med, col = color)
lines(x = X, y = x_med, col = color)
lines(x = X, y = x_low, col = color)
lines(x = X, y = x_upp, col = color)




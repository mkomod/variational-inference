library(ggplot2)
library(StanHeaders)
library(rstan)
library(here)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
# dont remember what this does: rstan_option(auto_write = TRUE)

# Simulate dataset:
set.seed(12)
X <- seq(0, 10, length.out=20)
K_ <- kern.gauss(X, X, l = 2, s = 3)
f <- mvtnorm::rmvnorm(1, mean=rep(0, length(X)),  K_) 
Y <- f  + rlaplace(length(X), eta=2)
round(Y,2)
bounds <- quantile(rlaplace(1000, eta=1.5), 0.025)
lower <- f + bounds[1]
upper <- f - bounds[1]
plot(X, f, ylim=c(-15, 15), type="b", pch=16, cex=0.7)
lines(X, upper, col="red")
lines(X, lower, col="red")

points(X, Y, pch=17, col="green")



#Fit the model in stan given the true parameters.

x2 <- 0
pred_data = list(  N1= length(X),
                   x1 = X,
                   y1 = as.vector(Y),
                   N2 = 1,
                   x2 = as.array(10.1)
  )


# used code from stan's youtube
pred_fit = stan(file = "FixedHyperHMC.stan",data = pred_data, chains = 4, iter = 2000)

if(!dir.exists(here("R/gpvi/results"))) {
  dir.create(here("R/gpvi/results"))
}

saveRDS(
  pred_fit, 
  file = here(paste0("R/gpvi/results/", "fit-", lrhoeta341point5, ".rds"))
)

print(pred_fit)
sim_summary = as.data.frame(summary(pred_fit)[[1]])

x_low = sim_summary$`2.5%`[17:(17+length(X)-1)]
x_med = sim_summary$`50%`[17:(17+length(X)-1)]
x_upp = sim_summary$`97.5%`[17:(17+length(X)-1)]

x_med

color = "blue"
points(x = X, y = x_med, col = color)
lines(x = X, y = x_med, col = color)
lines(x = X, y = x_low, col = color)
lines(x = X, y = x_upp, col = color)

library(shiny)
library(shinystan)
my_shinystan <- as.shinystan(pred_fit)
launch_shinystan(my_shinystan)




# Fit model with priors on hyperparameters:

print(pred_fit)
sim_summary = as.data.frame(summary(pred_fit)[[1]])

x_low = sim_summary$`2.5%`[20:(20+length(X)-1)]
x_med = sim_summary$`50%`[20:(20+length(X)-1)]
x_upp = sim_summary$`97.5%`[20:(20+length(X)-1)]

x_med

color = "blue"
points(x = X, y = x_med, col = color)
lines(x = X, y = x_med, col = color)
points(x = X, y = x_low, col = color)
points(x = X, y = x_upp, col = color)



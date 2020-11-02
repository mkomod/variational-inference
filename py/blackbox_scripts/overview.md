# **Generative Model**

Body fat dataset.
- Y is the bodyfat
- X are the predictors
- The mean has been removed from the dataset.

Y | (beta,sigma) ~ Normal(X * beta, sigma)

beta_i ~ Normal(0, 1), i=1,...,D=13 

sigma ~ Gamma(.5, .5)

# **Two blackbox methods**

## *KLVI (maximise the lower bound)*
- I posited a normal mean-field with dimension D+1.

- khat approx 0.9.

- Correlation in data -> mean-field is perhaps unsuitable.

## *CHIVI (minimise the upper bound (alpha=2))*
- I posited a normal mean-field with dimension D+1.

- khat approx 2.

    - k should be bounded by 1, values greater than 1 indicates poor finite sample performance.

    - Potential issues in Monte Carlo approximation of the CUBO objective and corresponding gradient is highlighted in https://arxiv.org/abs/1611.00328.


# **VSBC**
- Number of replications: M = 100.
    - Maybe too small.
- Compared KLVI and CHIVI algorithm.
    - One run of KLVI approx 20-30secs.
    - One run of CHIVI approx 40-60secs.
- No intercept included (I included at intercept in now and rerunning this test, will be done in few hours).
- Generated mc_samples = 100000 Monte Carlo samples.
- p-values computed using Monte Carlo Samples with different importance weights w i.e. w=1, w=SISweights and w = PSISweights
- Not much improvement in histograms observed by using the different weights.


### VI inference methods: 
1. MF_VI:=Mean field (independent gaussian Variational family)  stocahastic variational inference
2. 2. MVN_VI: Multivariate Gaussian variational family stocahastic variational inference

### Data: 
Bodymass dataset (dropped density as that's an expensive feature to obtain and the point of this exercise is to develop a cheap way to measure bodymass)

- 13 features

- Y="Bodymaxx"

### Priors: 
- N(0, 2) on weights; 
- N(0, 1) on bias; 
- HalfCauchy(1.0) on sigma


### Inference details: 

- VI: each of the two methods (MF_VI and MVT_VI)
 - optimse over ~5000 steps (calculation takes about 10-15second) 
 - adam optimiser with lr=0.009.

- HMC: 1000 samples, 200 burn-in; takes about 1m30sec

### VSBC diagnostic

- Using the variational posterior

- Using SNIS smoothed samples from the variational posterior

- Using PSIS smoothed samples from the variational posterior.

p-values are calculated using 500 runs of the Algorithm (draw from prior -> simulate data -> run VI -> calculate p-value)

Generally the variational posterior seems unbiased, except for log-sigma, so the smoothing does not do much.



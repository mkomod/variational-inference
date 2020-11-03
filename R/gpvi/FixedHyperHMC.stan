//taken from https://www.youtube.com/watch?v=132s2B-mzBg, suggested 2000 iter at least.
data {
  int<lower=1> N1; //number data observed 
  real x1[N1];
  vector[N1] y1;
  int<lower=0> N2; //number data to be predicted 
  real x2[N2];
  //Hyperparameters
  //real<lower = 0> rho;
  //real<lower = 0> alpha;
  //real<lower = 0> eta_inv;
  }
  
transformed data {
  int<lower=1> N = N1+N2;
  real x[N]; // number all
  for (n1 in 1:N1) x[n1] = x1[n1];
  for (n2 in 1:N2) x[N1+n2] = x2[n2];
}

parameters {
  vector[N] a;
}

transformed parameters{
  vector[N] f;
  {
    matrix[N,N] K = cov_exp_quad(x, 3, 2) + diag_matrix(rep_vector(1e-10, N));
    matrix[N,N] L_K = cholesky_decompose(K);
    f = L_K * a;
  }
}

model {
  a ~ std_normal();
  
  y1 ~ double_exponential(f[1:N1], 1.0/1.5);
    //for (i in 1:N){
        //aiming for laplace
        //y[i] ~ normal(f[i], 2/eta^2);
    //}
}
generated quantities{
  vector[N2] y2;
  for (n2 in 1:N2)
    y2[n2] = double_exponential_rng(f[N1 + n2], 1.0/1.5);
}
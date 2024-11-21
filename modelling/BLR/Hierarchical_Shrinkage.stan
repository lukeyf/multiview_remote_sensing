/*
Adapting the code for Hierarchical Shrinkage prior discussed here: https://arxiv.org/abs/1508.02502
*/
data{
    int<lower=1> N;
    int<lower=1> n;
    int<lower=1> B;
    vector[N] Y;
    matrix[N,B] X;
    //matrix[n,B] Xts ;
    real<lower=1> nu;  // give nu =  1 for horseshoe prior
}
transformed data {
    real delta = 1e-9;
}
parameters {
    real w0;
    real<lower=0> sigma;
    vector[B] z;
    real<lower=0> r1_global;
    real<lower=0> r2_global;
    vector<lower=0>[B] r1_local;
    vector<lower=0>[B] r2_local;
}
transformed parameters {
    real<lower=0> tau;
    vector<lower=0>[B] lambda;
    vector[B] w;
    tau = r1_global * sqrt(r2_global);
    lambda = r1_local .* sqrt(r2_local);
    w = z .* lambda*tau;
}
model{
    target += normal_lpdf(Y|w0 + X*w, sigma);
    // half t-prior for lambdas
    target += normal_lpdf(z|0,1);
    
    target += normal_lpdf(w0|0,5);
    target += normal_lpdf(r1_local|0.0, 1.0);
    target += inv_gamma_lpdf(r2_local|0.5*nu, 0.5*nu);

    target += normal_lpdf(r1_global|0.0, 1.0);
    target += inv_gamma_lpdf(r2_global|0.5, 0.5);
    //for sigma, flat prior
}
/*
Adapting the code for Hierarchical Shrinkage prior discussed here: https://arxiv.org/abs/1508.02502
*/
data{
    int<lower=1> N;
    int<lower=1> B;
    vector[N] Y;
    matrix[N,B] X;
    real<lower=1> nu_global;  
    real<lower=1> nu_local;
    real<lower=0> scale_icept;
    real<lower=0> scale_global;
    real<lower=0> slab_scale;
    real<lower=0> slab_df;
}
transformed data {
    real delta = 1e-9;
}
parameters {
    real w0;
    real log_sigma;
    vector[B] z;
    real<lower=0> r1_global;
    real<lower=0> r2_global;
    vector<lower=0>[B] r1_local;
    vector<lower=0>[B] r2_local;
    real<lower=0> caux;
}
transformed parameters {
    real<lower=0> sigma;
    real<lower=0> tau;
    vector[B] w;
    sigma = exp(log_sigma);
    tau = r1_global * sqrt(r2_global)*scale_global*sigma;
    {
        vector[B] lambda;   
        vector[B] lambda_tilde;
        real c;
        lambda = r1_local .* sqrt(r2_local);
        c = slab_scale * sqrt(caux);
        lambda_tilde = sqrt(square(c) * square(lambda) ./ (square(c) + square(tau)*square(lambda)) );
        w = z .* lambda_tilde*tau;
    } 
}
model{
    target += normal_lpdf(Y|w0 + X*w, sigma);
    target += normal_lpdf(w0|0,5);
    target += normal_lpdf(z|0,1);    
    target += normal_lpdf(r1_local|0.0, 1.0);
    target += inv_gamma_lpdf(r2_local|0.5*nu_local, 0.5*nu_local);

    target += normal_lpdf(r1_global|0.0, 1.0);
    target += inv_gamma_lpdf(r2_global|0.5*nu_global, 0.5*nu_global);

    target += inv_gamma_lpdf(caux|0.5* slab_df , 0.5* slab_df);
}
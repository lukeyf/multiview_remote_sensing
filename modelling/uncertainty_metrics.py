# Uncertainty metrics 

import numpy as np
from scipy.stats import norm
from math import sqrt, pi
import pandas as pd


def interval_length_coverage(y_true, y_pred, alpha=0.95):
    """
    Compute interval length and coverage of the interval given quantile level alpha
    The higher the coverage, the better the interval captures the true value
    The lower the interval length, the better the interval is
    
    Parameters
    ----------
    y_true : np.ndarray
        n_test x 1, True values
    y_pred : np.ndarray
        n_test x n_predict_samples, Predicted samples from posterior distribution
        
    Returns
    -------
    interval_length : float
        Average interval length
    coverage : float
        Coverage of the interval
    """
    y_pred = np.quantile(y_pred, [0.5 - alpha / 2, 0.5 + alpha / 2], axis=1).T
    interval_length = y_pred[:, 1] - y_pred[:, 0]
    coverage = np.mean((y_true >= y_pred[:, 0]) & (y_true <= y_pred[:, 1]))
    return np.mean(interval_length), coverage


def interval_length_coverage_normal(y_true, mu, sigma, alpha=0.95):
    """
    Compute interval length and coverage of the interval given quantile level alpha
    for a Gaussian distribution parameterized by mu and sigma.
    
    Parameters
    ----------
    mu : np.ndarray
        n_test x 1, Mean values of the Gaussian predictive distribution
    sigma : np.ndarray
        n_test x 1, Standard deviation values of the Gaussian predictive distribution
    y_true : np.ndarray
        n_test x 1, True values
    alpha : float, optional
        Quantile level for the interval (default is 0.95)
        
    Returns
    -------
    interval_length : float
        Average interval length
    coverage : float
        Coverage of the interval
    """
    # Compute the quantiles for the interval
    lower_quantile = norm.ppf(0.5 - alpha / 2, loc=mu, scale=sigma)
    upper_quantile = norm.ppf(0.5 + alpha / 2, loc=mu, scale=sigma)
    
    # Calculate interval length
    interval_length = upper_quantile - lower_quantile
    
    # Calculate coverage
    coverage = np.mean((y_true >= lower_quantile) & (y_true <= upper_quantile))
    
    return np.mean(interval_length), coverage
    

def crps_norm(x, mu, sigma):
    """
    Compute CRPS score for Gaussian distribution
    
    Parameters
    ----------
    x : np.ndarray
        n_test x 1, True values
    mu : np.ndarray
        n_test x 1, Predicted mean
    sigma : np.ndarray
        n_test x 1, Predicted standard deviation
    
    Returns
    -------
    crps : float
        CRPS score
    """
    
    sx = (x-mu)/sigma
    return np.mean(sigma * (sx*(2*norm.cdf(sx)-1) + 2 * norm.pdf(sx) - 1/sqrt(pi)))

def NLL(y_true, y_pred_mean, y_pred_std):
    """
    Compute negative log likelihood, assuming Gaussian distribution
    The lower the NLL, the better the model
    
    Parameters
    ----------
    y_true : np.ndarray
        n_test x 1, True values
    y_pred_mean : np.ndarray
        n_test x 1, Predicted mean
    y_pred_std : np.ndarray
        n_test x 1, Predicted standard deviation
        
    Returns
    -------
    nll : float
        Negative log likelihood
    """
    
    nll = np.mean(0.5 * np.log(2 * np.pi * y_pred_std**2) + 0.5 * ((y_true - y_pred_mean) / y_pred_std)**2)
    return nll

def test():
    # test
    y_true = np.random.randn(100)
    y_pred = np.random.normal(y_true[:, None], 1, (100, 1000))
    interval_length, coverage = interval_length_coverage(y_true, y_pred)
    _crps_score = crps_norm(y_true, y_pred.mean(axis=1), y_pred.std(axis=1))
    _NLL = NLL(y_true, y_pred.mean(axis=1), y_pred.std(axis=1))
    
    print('Pred std 1')
    print(f"Interval length: {interval_length}, Coverage: {coverage}")
    print(f"CRPS score: {_crps_score}")
    print(f"NLL: {_NLL}")
    
    y_pred = np.random.normal(y_true[:, None], 0.1, (100, 1000))
    interval_length, coverage = interval_length_coverage(y_true, y_pred)
    _crps_score = crps_norm(y_true, y_pred.mean(axis=1), y_pred.std(axis=1))
    _NLL = NLL(y_true, y_pred.mean(axis=1), y_pred.std(axis=1))
    
    print('Pred std 0.1')
    print(f"Interval length: {interval_length}, Coverage: {coverage}")
    print(f"CRPS score: {_crps_score}")
    print(f"NLL: {_NLL}")


def KidPovertyScores(label_path, pred_path):
    df = pd.read_csv(label_path)
    print(df.shape)
    print(df.head())
    target = df['target'].values
    post_mean = df['post_mean'].values
    post_std = df['post_sd'].values
    
    df = pd.read_csv(pred_path)
    post_samples = np.array(df.iloc[:, 1:].values).T
    print(post_samples.shape)
    
    interval_length, coverage = interval_length_coverage(target, post_samples)
    
    _crps_score = crps_norm(target, post_mean, post_std)
    
    _NLL = NLL(target, post_mean, post_std)
    
    print(f"Interval length: {interval_length}, Coverage: {coverage}")
    print(f"CRPS score: {_crps_score}")
    print(f"NLL: {_NLL}")
    
def KidPovertyScoresMCMC(label_path, pred_path):
    df = pd.read_csv(label_path)
    print(df.shape)
    print(df.head())
    target = df['target'].values
    post_mean = df['post_mean'].values
    post_std = df['post_sd'].values
    
    df = pd.read_csv(pred_path)
    post_samples = np.array(df.iloc[:, 1:].values)
    print(post_samples.shape)
    
    interval_length, coverage = interval_length_coverage(target, post_samples)
    
    _crps_score = crps_norm(target, post_mean, post_std)
    
    _NLL = NLL(target, post_mean, post_std)
    
    print(f"Interval length: {interval_length}, Coverage: {coverage}")
    print(f"CRPS score: {_crps_score}")
    print(f"NLL: {_NLL}")
    
    
    
def BaselineScores(label_path, pred_path):  
    pred_df = pd.read_csv(label_path)
    post_mean, post_std = pred_df.iloc[:,0].values, np.sqrt(pred_df.iloc[:,1].values)
    
    target = pd.read_csv(pred_path).values
    
    interval_length, coverage = interval_length_coverage_normal(target, post_mean, post_std)
    
    _crps_score = crps_norm(target, post_mean, post_std)
    
    _NLL = NLL(target, post_mean, post_std)
    
    print(f"Interval length: {interval_length}, Coverage: {coverage}")
    print(f"CRPS score: {_crps_score}")
    print(f"NLL: {_NLL}")
    
    
    
if __name__ == "__main__":
    test()
import numpy as np
from scipy.stats import multivariate_normal

class RegularizedDiscriminantAnalysis:
    def __init__(self, alpha=0.5, gamma=0.5):
        self.alpha = alpha
        self.gamma = gamma
        self.classes_ = None
        self.covariances_ = None
        self.means_ = None
        self.priors_ = None
    
    def fit(self, X, y):
        self.classes_, y_counts = np.unique(y, return_counts=True)
        self.priors_ = y_counts / len(y)
        self.means_ = [X[y == c].mean(axis=0) for c in self.classes_]
        # print(self.priors_)
        
        pooled_cov = sum(np.cov(X[y == c].T, bias=True) * (len(X[y == c]) - 1) for c in self.classes_) / (len(X) - len(self.classes_))
        
        self.covariances_ = [self.alpha * np.cov(X[y == c].T, bias=True) + (1 - self.alpha) * pooled_cov for c in self.classes_]
        self.covariances_ = [self.gamma * cov + (1 - self.gamma) * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0] for cov in self.covariances_]
    
    def predict_proba(self, X):
        # print(self.means_[0], self.covariances_[0], self.priors_[0])
        scores = np.stack([multivariate_normal.pdf(X, mean=mean, cov=cov) * prior
                           for mean, cov, prior in zip(self.means_, self.covariances_, self.priors_)], axis=1)
        return scores / scores.sum(axis=1, keepdims=True)
    
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
    
    def compute_log_likelihood(self, X, y):
        proba = self.predict_proba(X)
        # print(proba)
        y_encoded = np.eye(proba.shape[1])[y]
        proba = np.nan_to_num(proba, nan=1.0, posinf=1.0, neginf=1e-9)
        nll = -np.sum(y_encoded * np.log(np.clip(proba, 1e-9, 1-1e-9)))
        return nll

    def compute_aic(self, X, y):
        log_likelihood = self.compute_log_likelihood(X, y)
        # Number of parameters: sum of the number of elements in means and covariances
        num_params = sum(mean.size + cov.size for mean, cov in zip(self.means_, self.covariances_))
        print(num_params)
        aic = 2 * num_params - 2 * log_likelihood
        return aic

    def compute_bic(self, X, y):
        log_likelihood = self.compute_log_likelihood(X, y)
        num_params = sum(mean.size + cov.size for mean, cov in zip(self.means_, self.covariances_))
        bic = np.log(X.shape[0]) * num_params - 2 * log_likelihood
        return bic

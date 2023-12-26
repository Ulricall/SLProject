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
        print(self.classes_, y_counts)
        
        pooled_cov = sum(np.cov(X[y == c].T, bias=True) * (len(X[y == c]) - 1) for c in self.classes_) / (len(X) - len(self.classes_))
        
        self.covariances_ = [(1 - self.gamma) * np.cov(X[y == c].T, bias=True) + self.gamma * pooled_cov for c in self.classes_]
        self.covariances_ = [(1 - self.alpha) * cov + self.alpha * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0] for cov in self.covariances_]
    
    def predict_proba(self, X):
        scores = np.stack([multivariate_normal.pdf(X, mean=mean, cov=cov) * prior
                           for mean, cov, prior in zip(self.means_, self.covariances_, self.priors_)], axis=1)
        return scores / scores.sum(axis=1, keepdims=True)
    
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
    
    def compute_log_likelihood(self, X, y):
        proba = self.predict_proba(X)
        classes = self.classes_[np.argmax(proba, axis=1)]
        log_likelihood = np.sum(np.log(proba[np.arange(len(X)), classes == y]))
        return log_likelihood

    def compute_aic(self, X, y):
        log_likelihood = self.compute_log_likelihood(X, y)
        # Number of parameters: sum of the number of elements in means and covariances
        num_params = sum(mean.size + cov.size for mean, cov in zip(self.means_, self.covariances_))
        aic = 2 * num_params - 2 * log_likelihood
        return aic

    def compute_bic(self, X, y):
        log_likelihood = self.compute_log_likelihood(X, y)
        num_params = sum(mean.size + cov.size for mean, cov in zip(self.means_, self.covariances_))
        bic = np.log(X.shape[0]) * num_params - 2 * log_likelihood
        return bic

import numpy as np
from cvxopt import matrix, solvers

class BinarySVM:
    def __init__(self, C=1.0):
        self.C = C
        self.alpha = None
        self.w = None
        self.b = None

    def fit(self, X, y):
        m, n = X.shape
        y = y.reshape(-1, 1) * 1.0
        X_dash = y * X
        Q = np.dot(X_dash, X_dash.T) * 1.0

        P = matrix(Q)
        q = matrix(-np.ones((m, 1)))
        G = matrix(np.vstack((-np.eye(m), np.eye(m))))
        h = matrix(np.hstack((np.zeros(m), np.ones(m) * self.C)))
        A = matrix(y.reshape(1, -1))
        b = matrix(np.zeros(1))
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(solution['x'])

        sv = alphas > 1e-4
        self.alpha = alphas[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        
        self.w = np.sum(self.alpha * self.sv_y * self.sv, axis=0).reshape(-1, 1)
        self.b = np.mean(self.sv_y - np.dot(self.sv, self.w))

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))

class MultiClassSVM:
    def __init__(self, C=1.0):
        self.C = C
        self.classifiers = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for i in self.classes_:
            svm = BinarySVM(self.C)
            y_binary = np.where(y == i, 1, -1)
            svm.fit(X, y_binary)
            self.classifiers.append(svm)

    def predict(self, X):
        decision_scores = np.array([svm.decision_function(X) for svm in self.classifiers])
        return self.classes_[np.argmax(decision_scores, axis=0)]

import numpy as np
import sys

class logistic_regression(object):
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.W = None

    def fit_BGD(self, X, y):
        """Train logistic regression model with Batch Gradient Descent."""
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        for _ in range(self.max_iter):
            gradient = np.mean([self._gradient(xi, yi) for xi, yi in zip(X, y)], axis=0)
            self.W -= self.learning_rate * gradient
        return self

    def fit_SGD(self, X, y):
        """Train logistic regression model with Stochastic Gradient Descent."""
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        for _ in range(self.max_iter):
            indices = np.random.permutation(n_samples)
            for i in indices:
                gradient = self._gradient(X[i], y[i])
                self.W -= self.learning_rate * gradient
        return self

    def fit_miniBGD(self, X, y, batch_size):
        """Train logistic regression model with Mini-Batch Gradient Descent."""
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        for _ in range(self.max_iter):
            indices = np.random.permutation(n_samples)
            for i in range(0, n_samples, batch_size):
                X_batch = X[indices[i:i+batch_size]]
                y_batch = y[indices[i:i+batch_size]]
                gradient = np.mean([self._gradient(xi, yi) for xi, yi in zip(X_batch, y_batch)], axis=0)
                self.W -= self.learning_rate * gradient
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy loss for one sample."""
        z = _y * np.dot(self.W, _x)
        sigma = 1.0 / (1.0 + np.exp(-z))
        return (sigma - 1) * _y * _x

    def predict_proba(self, X):
        """Predict class probabilities for samples in X."""
        scores = X.dot(self.W)
        prob_positive = 1.0 / (1.0 + np.exp(-scores))
        prob_negative = 1 - prob_positive
        return np.column_stack((prob_positive, prob_negative))

    def predict(self, X):
        """Predict class labels for samples in X."""
        prob_positive = self.predict_proba(X)[:, 0]
        return np.where(prob_positive >= 0.5, 1, -1)

    def score(self, X, y):
        """Compute the mean accuracy on the given test data and labels."""
        preds = self.predict(X)
        return np.mean(preds == y)

    def get_params(self):
        """Get model parameters."""
        return self.W

    def assign_weights(self, weights):
        """Assign weights to the model."""
        self.W = weights
        return self
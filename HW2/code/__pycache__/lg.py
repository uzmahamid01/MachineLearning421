import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
	
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit_BGD(self, X, y):
        """Train perceptron model on data (X,y) with Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        n_samples, n_features = X.shape

		### YOUR CODE HERE
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)  # Initialize weights

        for _ in range(self.max_iter):
            gradient = np.zeros_like(self.W)
            for i in range(n_samples):
                gradient += self._gradient(X[i], y[i])
            
            # Update weights
            self.W -= self.learning_rate * gradient / n_samples
		### END YOUR CODE
        return self

    def fit_miniBGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)  # Initialize weights

        for _ in range(self.max_iter):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                gradient = np.zeros_like(self.W)
                for j in range(len(batch_X)):
                    gradient += self._gradient(batch_X[j], batch_y[j])
                
                self.W -= self.learning_rate * gradient / batch_size
		### END YOUR CODE
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with Stochastic Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)  # Initialize weights

        for _ in range(self.max_iter):
            for i in range(n_samples):
                gradient = self._gradient(X[i], y[i])
                self.W -= self.learning_rate * gradient
		### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        z = np.dot(self.W, _x)  # w^T x
        prediction = 1 / (1 + np.exp(-z))  # Sigmoid function θ(w^T x)
        
        # Compute gradient
        gradient = (prediction - _y) * _x
        return gradient
		### END YOUR CODE

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
		### YOUR CODE HERE
        z = np.dot(X, self.W)
        probabilities = 1 / (1 + np.exp(-z))
        return np.vstack([1 - probabilities, probabilities]).T
        ### END YOUR CODE


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
		### YOUR CODE HERE
        proba = self.predict_proba(X)
        return np.where(proba[:, 1] >= 0.5, 1, -1)
		### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
		### YOUR CODE HERE
        predictions = self.predict(X)
        return np.mean(predictions == y)
		### END YOUR CODE
    
    def assign_weights(self, weights):
        self.W = weights
        return self


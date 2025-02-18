

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Fri Sep  6 12:00:48 2019

# @author: Uzma Hamid
# """

import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression_multiclass(object):	
    def __init__(self, learning_rate, max_iter, k):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k = k 
        
        
    def fit_miniBGD(self, X, labels, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch GD.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,..,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """

		### YOUR CODE HERE
        n_samples, n_features = X.shape
        self.W = np.zeros((n_features, self.k))  

        labels = labels.astype(int)
        one_hot_labels = np.eye(self.k)[labels]

        for i in range(self.max_iter):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = one_hot_labels[indices]

            for j in range(0, n_samples, batch_size):
                X_batch = X_shuffled[j:j + batch_size]
                y_batch = y_shuffled[j:j + batch_size]
                gradients = np.zeros_like(self.W)
                for _x, _y in zip(X_batch, y_batch):
                    gradients += self._gradient(_x, _y)

                self.W -= self.learning_rate * gradients / batch_size

        return self

		### END YOUR CODE

    

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: One_hot vector. 

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        scores = np.dot(_x, self.W)
        exp_scores = np.exp(scores - np.max(scores))  
        probs = exp_scores / np.sum(exp_scores)
        
        # Gradient of cross-entropy
        _g = np.outer(_x, (probs - _y))
        return _g

		### END YOUR CODE
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        ### You must implement softmax by youself, otherwise you will not get credits for this part.

		### YOUR CODE HERE
        exp_scores = np.exp(x - np.max(x))  
        return exp_scores / np.sum(exp_scores, axis=0)
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


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 0,..,k-1.
        """
		### YOUR CODE HERE
        scores = np.dot(X, self.W)
        preds = np.argmax(scores, axis=1)
        return preds

		### END YOUR CODE


    def score(self, X, labels):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,]. Only contains 0,..,k-1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. labels.
        """
		### YOUR CODE HERE
        preds = self.predict(X)
        return np.mean(preds == labels)
		### END YOUR CODE


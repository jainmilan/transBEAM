import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.base import clone
from sklearn.utils.validation import _num_samples

import json
from . import models

class InstanceTransfer():
	def __init__(self, n_iterations=500, estimator=models.RandomForest().model):
		self.n_iterations = n_iterations
		self.estimator = estimator
		self.trained_model = None

	def fit(self, X_source, y_source, X_target, y_target):
		n_source = X_source.shape[0]
		n_target = X_target.shape[0]

		n_samples = n_source + n_target

		X = pd.concat([X_source, X_target])
		y = pd.concat([y_source, y_target])

		init_weights=None
		if init_weights is None:
			init_weights = np.ones(n_samples)
		else:
			assert _num_samples(init_weights) == n_samples

		P_values = np.empty((self.n_iterations, n_samples))

		error = np.empty(self.n_iterations)
		beta_0 = 1 / (1 + np.sqrt(2 * np.log(n_source / self.n_iterations)))
		beta_t = np.empty(self.n_iterations)

		weights = init_weights

		for _iter in np.arange(self.n_iterations):
			P_values[_iter] = weights / sum(weights)

			self.trained_model = clone(self.estimator).fit(X, y, sample_weight=P_values[_iter])
			y_pred_target = self.trained_model.predict(X_target)

			error[_iter] = np.sum(weights[n_source:] * np.abs(y_pred_target - y_target) / np.sum(weights[n_source:]))
			if error[_iter] > 0.5 or error[_iter] == 0:
				return self.trained_model
			    
			beta_t[_iter] = error[_iter] / (1 - error[_iter])
			
			# Update the new weight vector
			if _iter < self.n_iterations - 1:
			    y_pred_source = self.trained_model.predict(X_source)
			    weights[:n_source] = weights[:n_source] * (beta_0 ** np.abs(y_pred_source - y_source.values))
			    weights[n_source:] = weights[n_source:] * (beta_t[_iter] ** -np.abs(y_pred_target - y_target.values))

		self.weights = weights
		self.P_values = P_values
		self.error = error
		self.beta_0 = beta_0
		self.beta_t = beta_t

	def predict(self, X_test, prefix=""):
		assert self.trained_model, "No trained model found, either train the model or load a trained model"

		# predict and return
		y_pred = self.trained_model.predict(X_test)

		# convert predictions to dataframe
		y_test = pd.DataFrame(y_pred, index=X_test.index, columns=['predictions'])
		# return the predictions
		return y_test
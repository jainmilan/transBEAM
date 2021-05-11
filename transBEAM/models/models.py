import pandas as pd

import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor

class RandomForest():
	def __init__(self):
		self.model = RandomForestRegressor(random_state=42)
		self.trained = False

	def train(self, X, y, save_dir=None, prefix=""):
		# train the model
		self.model.fit(X, y)
		self.trained = True

		# if asked, save the model
		if save_dir is not None: dump(self.model, save_dir + prefix + '_rf.joblib')

	def predict(self, X_test, prefix=""):
		assert self.trained, "No trained model found, either train the model or load a trained model"

		# predict and return
		y_pred = self.model.predict(X_test)

		# convert predictions to dataframe
		y_test = pd.DataFrame(y_pred, index=X_test.index, columns=['predictions'])
		# return the predictions
		return y_test

class FeedForwardNetwork():
	def __init__(self, input_shape):
		self.model = tf.keras.Sequential([
				tf.keras.layers.InputLayer(input_shape=(input_shape,)),
				tf.keras.layers.Dense(256, activation='relu'),
				tf.keras.layers.Dropout(0.5),
				tf.keras.layers.Dense(256, activation='relu'),
				tf.keras.layers.Dense(1)
			])
		self.trained = False
	
	def train(self, X, y, val_split=None, prefix="", save_dir=None):
		# cast data
		X_casted = tf.cast(X, tf.float32)
		y_casted = tf.cast(y, tf.float32)

		self.model.compile(optimizer='adam', loss='mae')

		self.model.fit(X_casted, y_casted,
			epochs=50,
			shuffle=True,
			validation_split=val_split
		)

		self.trained = True

		# if asked, save the model
		if save_dir is not None: self.model.save(save_dir + prefix + '_nn.h5')

	def predict(self, X_test, prefix=""):
		assert self.trained, "No trained model found, either train the model or load a trained model"

		# cast the input
		X_casted = tf.cast(X_test, tf.float32)

		# predict and return
		y_pred = self.model.predict(X_casted)

		# convert predictions to dataframe
		y_test = pd.DataFrame(y_pred, index=X_test.index, columns=['predictions'])
		# return the predictions
		return y_test



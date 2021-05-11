import os

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class DataPreprocessor():
	def __init__(self):
		self.scaler = None

	def data_split(self, data, train_size, val_size=0.0, shuffle=True, random_state=12, save_dir=None):
		'''
			Split the data.
		'''
		assert data is not None, "No data found, you will need to load_data first."
		assert np.abs(train_size) <= 1, "Invalid split of data, size of training data should lie between 0 and 1." 

		print("---------------- Size of Training Data: ", train_size, "% -------------------------")

		# split the data
		test_size = 1 - train_size
		
		train_data, test_data = train_test_split(data, shuffle=shuffle, random_state=random_state, test_size=test_size)
		if val_size > 0:
			train_data, val_data = train_test_split(train_data, shuffle=shuffle, random_state=random_state, test_size=val_size)

		print("Size of Training Data: ", train_data.shape)
		if val_size > 0:
			print("Size of Validation Data: ", val_data.shape)
		print("Size of Testing Data: ", test_data.shape)

		if save_dir:
			split_directory = save_dir + 's' + str(np.round(train_size, 2)) + '_' \
											+ str(np.round(val_size, 2)) + '_' \
											+ str(np.round(test_size, 2)) + '/'											
			if not os.path.exists(split_directory):
				os.makedirs(split_directory)

			print("Data Directory: ", split_directory)
			
			train_data.to_csv(split_directory + 'train.csv')
			if val_size > 0:
				val_data.to_csv(split_directory + 'val.csv')
			test_data.to_csv(split_directory + 'test.csv')

		if val_size > 0:
			return [train_data, val_data, test_data]
		else:
			return [train_data, test_data]

	def season_wise_split(self, data, train_season, val_size=0.0, shuffle=True, random_state=12, save_dir=None):
		'''
			Split the data.
		'''
		assert data is not None, "No data found, you will need to load_data first."
		
		print("---------------- Training Season: ", train_season, "% -------------------------")

		# split the data
		seasons = {'summer': [6, 7, 8], 'winter': [12, 1, 2], 'fall': [9, 10, 11], 'spring': [3, 4, 5]}
		season_names = seasons.keys()
		
		c = set()
		c.add(train_season)

		b = set(season_names).difference(c)
		testing_months = []
		for s in b:
			testing_months += seasons[s]

		train_data = data[data.index.month.isin(seasons[train_season])].dropna().copy()
		test_data = data[data.index.month.isin(testing_months)].dropna().copy()

		# validation data if exists
		if val_size > 0:
			train_data, val_data = train_test_split(train_data, shuffle=shuffle, random_state=random_state, test_size=val_size)

		print("Size of Training Data: ", train_data.shape)
		if val_size > 0:
			print("Size of Validation Data: ", val_data.shape)
		print("Size of Testing Data: ", test_data.shape)

		if save_dir:
			split_directory = save_dir + 's_' + train_season + '/'											
			if not os.path.exists(split_directory):
				os.makedirs(split_directory)

			print("Data Directory: ", split_directory)
			
			train_data.to_csv(split_directory + 'train.csv')
			if val_size > 0:
				val_data.to_csv(split_directory + 'val.csv')
			test_data.to_csv(split_directory + 'test.csv')

		if val_size > 0:
			return [train_data, val_data, test_data]
		else:
			return [train_data, test_data]

	def normalize_data(self, df):
		if self.scaler is not None:
			scaled_output = self.scaler.transform(df)
		else:
			self.scaler = preprocessing.MinMaxScaler()
			scaled_output = self.scaler.fit_transform(df)

		df_norm = pd.DataFrame(data=scaled_output, index=df.index, columns=df.columns)
		return df_norm

	def inverse_transform(self, df_norm, y_pred, y_col):
		df_norm[y_col] = y_pred
		original_output = self.scaler.inverse_transform(df_norm)

		df_act = pd.DataFrame(data=original_output, index=df_norm.index, columns=df_norm.columns)
		return df_act
import os
import numpy as np
import pandas as pd

from joblib import dump, load

from sklearn import preprocessing
from sklearn.model_selection import KFold, train_test_split

# from .. import utilities

class TimeSeriesDataHandler():
	def __init__(self):
		# initialize data handler
		self.data = dict()

	def add_data(self, data, name):
		# add new frame
		self.data[name] = data

	def get_data(self, name):
		# get data frame
		return self.data[name]

	def save_data(self, name, filepath):
		# save the processed data frame
		self.data[name].to_csv(filepath)

	def sel_columns(self, sel_cols):
		assert isinstance(sel_cols, list), "Expected a list of columns, found " + str(type(sel_cols))

		# select columns
		self.sel_data[name] = self.sel_data[name][sel_cols]

	def add_datetime(self, name, dt_cols, dt_format, dt_colname='datetime', set_index=True):
		# merge columns into a single date time 
		self.data[name][dt_colname] = self.data[name][dt_cols].astype(str).apply(' '.join, 1)

		# create datetime
		self.data[name][dt_colname] = pd.to_datetime(self.data[name][dt_colname], format=dt_format)

		# set datetime as index if required
		if set_index: self.data[name] = self.data[name].set_index(dt_colname)


	def add_static_features(self, name):
		'''
		'''

		assert self.data[name] is not None, "No data found, run get_data() to read the datafile."
		assert isinstance(self.data[name].index, pd.DatetimeIndex), "Need pandas DatetimeIndex to add static features."
		
		# Day of Week
		self.data[name]["DayOfWeek"] = self.data[name].index.dayofweek

		# Day Type - 1 for Weekday/ 0 for Weekend
		self.data[name]["Weekday"] = 1
		self.data[name].loc[self.data[name]["DayOfWeek"].isin([5, 6]), "Weekday"] = 0

		# Hour of the Day
		self.data[name]["HourOfDay"] = self.data[name].index.hour
		
		# Working/Non-Working Hours
		self.data[name]["WorkHour"] = 0

		# working hours
		indxs = self.data[name].index[self.data[name].index.indexer_between_time("09:00", "18:00")]
		self.data[name].loc[indxs, "WorkHour"] = 1

		# Month
		self.data[name]["Month"] = self.data[name].index.month

	def add_shift_cols(self, name, col_names, shift_by):
		'''
			params:
			col_names: name of the columns to shift
			shift_by: shift periods
		'''
		assert self.data[name] is not None, "No data found, run get_data() to read the datafile."
		assert isinstance(col_names, list), "Invalid data type for col_names, expecting list."
		assert isinstance(shift_by, list), "Invalid data type for shift_by, expecting list."
		assert len(col_names)==len(shift_by), "Length of column names and shift by column is a mismatch, must be of same lenghts."
		
		# shift columns
		for col_name, periods in zip(col_names, shift_by):
			assert col_name in self.data[name].columns, "{col_name} not found in data columns."
			self.data[name]["Prev" + col_name] = self.data[name][col_name].shift(periods=periods)

	def hot_encoding(self, name, col_name):
		# Hot Encoded
		self.data[name] = pd.concat([self.data[name], pd.get_dummies(self.data[name][col_name], prefix=col_name)], axis=1)		

	def dropna(self, name, how='any'):
		# Drop na from the data
		self.data[name] = self.data[name].dropna(how=how)

	def remove_columns_with_missing_data(self, name, threshold):

		print("Shape of the Feature Set (initially):", self.data[name].shape)

		# count missing values in each column
		ds_counts = self.data[name].isna().sum()

		# remove columns with more than 500 missing values
		features_wonacols = list(ds_counts[ds_counts < threshold * self.data[name].shape[0]].index.values)
		self.data[name] = self.data[name][features_wonacols].copy()

		print("Shape of the Feature Set (after removing columns with more than", threshold * self.data[name].shape[0], "missing values):", self.data[name].shape)

		# drop rows with atleast one missing value
		# self.sel_data = self.sel_data.dropna()
		# print("Shape of the Feature Set (after removing rows with atleast one missing values):", self.sel_data.shape)

	def remove_duplicate_indxs(self, name):
		# remove duplicate indexes
		self.data[name] = self.data[name][~self.data[name].index.duplicated(keep='last')]
		print("Shape of the Feature Set (after removing duplicate indexes):", self.data[name].shape)

class DataHandler():
	def __init__(self, data_dict):
		self.data_dict = data_dict

	def parse_data(self, dropna=True):
		'''
			Read the data file.
			params
			
		'''
		print("Loading data file.")

		# empty frame and its parameters dict
		self.data = None
		param_dict = dict()

		# check if datetime columns are set
		assert self.data_dict['datetime_cols'] is not None, "No datetime columns found, please set using set_datetime_cols()."

		# if user defined cols
		if self.data_dict['data_cols'] is not None:
			param_dict["usecols"] = self.data_dict['data_cols']
			print("Reading user-defined columns from the data file:", self.data_dict['data_cols'])
		else:
			print("Reading all the data columns.")
		
		# create dict to read the data file
		param_dict["filepath_or_buffer"] = self.data_dict['raw_datafile']

		# if datetime column name is given
		if len(self.data_dict['datetime_cols']) == 1:
			param_dict["index_col"] = self.data_dict['datetime_cols'][0]
			param_dict["parse_dates"] = True

			self.data = pd.read_csv(**param_dict)
		# if names of date and time columns are given
		elif len(self.data_dict['datetime_cols']) == 2:
			self.data = pd.read_csv(**param_dict)

			# add a datetime index
			self.data["DateTime"] = pd.to_datetime(self.data[self.data_dict['datetime_cols'][0]] + \
											"-2017 " + \
											self.data[self.data_dict['datetime_cols'][1]], \
											format=self.data_dict['datetime_format'])
			self.data = self.data.set_index("DateTime")
		# invalid datetime column list
		else:
			raise AssertionError("Invalid datetime column")

		self.data.to_csv(self.data_dict['loaded_datafile'])

	def load_data(self):
		
		# save the processed data frame
		self.data = pd.read_csv(self.data_dict['loaded_datafile'], index_col=[0], parse_dates=True)

	def load_processed_data(self):
		
		# save the processed data frame
		self.data = pd.read_csv(self.data_dict['processed_datafile'], index_col=[0], parse_dates=True)

	def save_data(self):
		
		# save the processed data frame
		self.data.to_csv(self.data_dict['processed_datafile'])

	def add_static_features(self):
		'''
		'''

		assert self.data is not None, "No data found, run get_data() to read the datafile."
		
		# Day of Week
		self.data["DayOfWeek"] = self.data.index.dayofweek

		# Day Type - 1 for Weekday/ 0 for Weekend
		self.data["Weekday"] = 1
		self.data.loc[self.data["DayOfWeek"].isin([5, 6]), "Weekday"] = 0

		# Hour of the Day
		self.data["HourOfDay"] = self.data.index.hour
		# Hot Encoded
		self.data = pd.concat([self.data, pd.get_dummies(self.data['HourOfDay'], prefix='Hour')], axis=1)

		# Working/Non-Working Hours
		self.data["WorkHour"] = 0

		# working hours
		indxs = self.data.index[self.data.index.indexer_between_time("09:00", "18:00")]
		self.data.loc[indxs, "WorkHour"] = 1

		# Month
		self.data["Month"] = self.data.index.month

	def shift_cols(self, col_names, shift_by):
		'''
			params:
			col_names: name of the columns to shift
			shift_by: shift periods
		'''
		assert self.data is not None, "No data found, run get_data() to read the datafile."
		assert isinstance(col_names, list), "Invalid data type for col_names, expecting list."
		assert isinstance(shift_by, list), "Invalid data type for shift_by, expecting list."
		assert len(col_names)==len(shift_by), "Length of column names and shift by column is a mismatch, must be of same lenghts."
		
		# shift columns
		for col_name, periods in zip(col_names, shift_by):
			assert col_name in self.data.columns, "{col_name} not found in data columns."
			self.data["Prev" + col_name] = self.data[col_name].shift(periods=periods)
	
	def dropna(self):
		self.data = self.data.dropna()

	def split_data(self, train_size, val_size=0.0, shuffle=True, random_state=12, save=True):
		'''
			Split the data.
		'''
		assert self.data is not None, "No data found, you will need to load_data first."
		assert np.abs(train_size) <= 1, "Invalid split of data, size of training data should lie between 0 and 1." 

		print("---------------- Size of Training Data: ", train_size, "% -------------------------")

		# split the data
		test_size = 1 - train_size
		
		train_data, test_data = train_test_split(self.data, shuffle=shuffle, random_state=random_state, test_size=test_size)
		if val_size > 0:
			train_data, val_data = train_test_split(train_data, shuffle=shuffle, random_state=random_state, test_size=val_size)

		print("Size of Training Data: ", train_data.shape)
		if val_size > 0:
			print("Size of Validation Data: ", val_data.shape)
		print("Size of Testing Data: ", test_data.shape)

		if save:
			split_directory = self.data_dict['model_folder'] + 's' + str(np.round(train_size, 2)) + '_' \
															   + str(np.round(val_size, 2)) + '_' \
															   + str(np.round(test_size, 2)) + '/'											
			if not os.path.exists(split_directory):
				os.makedirs(split_directory)

			print("Data Directory: ", split_directory)
			
			train_data.to_csv(split_directory + 'train.csv')
			if val_size > 0:
				val_data.to_csv(split_directory + 'val.csv')
			test_data.to_csv(split_directory + 'test.csv')

		return split_directory

	def season_wise_split(self, train_season, val_size=0.0, shuffle=True, random_state=12, save=True):
		'''
			Split the data.
		'''
		assert self.data is not None, "No data found, you will need to load_data first."
		
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

		train_data = self.data[self.data.index.month.isin(seasons[train_season])].dropna().copy()
		test_data = self.data[self.data.index.month.isin(testing_months)].dropna().copy()

		# validation data if exists
		if val_size > 0:
			train_data, val_data = train_test_split(train_data, shuffle=shuffle, random_state=random_state, test_size=val_size)

		print("Size of Training Data: ", train_data.shape)
		if val_size > 0:
			print("Size of Validation Data: ", val_data.shape)
		print("Size of Testing Data: ", test_data.shape)

		if save:
			split_directory = self.data_dict['model_folder'] + 's_' + train_season + '/'											
			if not os.path.exists(split_directory):
				os.makedirs(split_directory)

			print("Data Directory: ", split_directory)
			
			train_data.to_csv(split_directory + 'train.csv')
			if val_size > 0:
				val_data.to_csv(split_directory + 'val.csv')
			test_data.to_csv(split_directory + 'test.csv')

		return split_directory

	def get_split_dirs(self):
		assert len(self.data_dict['split_dirs'])>0, "No splits found, first run split_data() to get data splits."
		return self.data_dict['split_dirs']

	def set_x_cols(self, x_cols):
		assert isinstance(x_cols, list), "Invalid data type for x_cols, expecting list."
		self.data_dict['x_cols'] = x_cols
	
	def set_y_col(self, y_col):
		self.data_dict['y_col'] = y_col
	
	def split_X_y(self, split_dir, dtypes, normalize_data=True, save_data=True):
		
		assert self.data_dict["x_cols"] is not None, "No feature set found, set X columns using set_x_cols()."
		assert self.data_dict["y_col"] is not None, "No output column found, set output column using set_y_col()."
		
		# feature set and output variable
		X_cols = self.data_dict["x_cols"]
		y_col = self.data_dict["y_col"]

		n_features = len(X_cols)

		# for each split
		split_id = '/'.join(split_dir.split('/')[-4:])
		print("------- Splitting data for", split_id, "--------------")
		
		# dictionary to return
		data_split = dict()
		data_split["split_dir"] = split_dir

		for dtype in dtypes:
			data = pd.read_csv(split_dir + dtype + '.csv', index_col=[0], parse_dates=True)

			X = data[X_cols]
			y = data[y_col]

			if normalize_data:
				if dtype == "train":
					X_scaler = self.fit_scaler(X, split_dir, 'X')
					y_scaler = self.fit_scaler(y, split_dir, 'y')
				else:
					X_scaler = self.get_scaler(split_dir, 'X')
					y_scaler = self.get_scaler(split_dir, 'y')

				X = self.normalize(X, X_scaler)
				dfn_X = pd.DataFrame(data=X, index=data.index, columns=X_cols)
				if save_data:
					dfn_X.to_csv(split_dir + dtype + '_Xn.csv')
				
				y = self.normalize(y, y_scaler)
				dfn_y = pd.DataFrame(data=y, index=data.index, columns=y_col)
				if save_data:
					dfn_y.to_csv(split_dir + dtype + '_yn.csv')

			print("Shape of", dtype, " X is:", X.shape)
			print("Shape of", dtype, " y is:", y.shape)
			data_split[dtype] = {'X': X, 'y': y}

		return data_split
		'''
		# get training data
		train_data = pd.read_csv(split_dir + 'train.csv', index_col=[0], parse_dates=True)
		X_train = train_data[X_cols]
		y_train = train_data[y_col]

		if normalize_data:
			X_scaler = self.fit_scaler(X_train, split_dir, 'X')
			y_scaler = self.fit_scaler(y_train, split_dir, 'y')

			X_train = self.normalize(X_train, X_scaler)
			dfn_X_train = pd.DataFrame(data=X_train, index=train_data.index, columns=X_cols)
			if save_data:
				dfn_X_train.to_csv(split_dir + 'train_Xn.csv')
			
			y_train = self.normalize(y_train, y_scaler)
			dfn_y_train = pd.DataFrame(data=y_train, index=train_data.index, columns=y_col)
			if save_data:
				dfn_y_train.to_csv(split_dir + 'train_yn.csv')
		
		# get validation data
		val_data = pd.read_csv(split_dir + 'val.csv', index_col=[0], parse_dates=True)
		X_val = val_data[X_cols]
		y_val = val_data[y_col]

		if normalize_data:
			X_val = self.normalize(X_val, X_scaler)
			dfn_X_val = pd.DataFrame(data=X_val, index=val_data.index, columns=X_cols)
			if save_data:
				dfn_X_val.to_csv(split_dir + 'val_Xn.csv')
			
			y_val = self.normalize(y_val, y_scaler)
			dfn_y_val = pd.DataFrame(data=y_val, index=val_data.index, columns=y_col)
			if save_data:
				dfn_y_val.to_csv(split_dir + 'val_yn.csv')
	'''
	def get_scaler(self, split_dir, stype):
		if (stype == "X"):
			scaler = load(split_dir + 'X_scaler.joblib')
		elif (stype == "y"):
			scaler = load(split_dir + 'y_scaler.joblib')

		return scaler

	def fit_scaler(self, data, split_dir, stype, save_scaler=True):
		# check if data columns matches with stype
		scaler = preprocessing.MinMaxScaler()
		scaler = scaler.fit(data)

		assert stype in ["X", "y"], "Invalid data type, it should either be X or y."

		if (stype == "X") and save_scaler:
			dump(scaler, split_dir + 'X_scaler.joblib')
		elif (stype == "y") and save_scaler:
			dump(scaler, split_dir + 'y_scaler.joblib')

		return scaler

	def unnormalized(self, data, scaler):
		return scaler.inverse_transform(data)
	
	def normalize(self, data, scaler):
		return scaler.transform(data)

	def get_split_dir_data(self, info_dict, dtype):
		split_dir = utilities.get_processed_filepath(info_dict['building'], info_dict['split'])
		Xn = pd.read_csv(split_dir + dtype + '_Xn.csv', index_col=[0], parse_dates=True)
		yn = pd.read_csv(split_dir + dtype + '_yn.csv', index_col=[0], parse_dates=True)

		print(Xn.shape, yn.shape)

		data = {'split_dir': split_dir, dtype: {'X': Xn.values, 'y': yn.values}}

		return data
	
	
	
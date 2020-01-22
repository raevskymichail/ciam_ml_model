import pandas as pd
import numpy as np
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, GroupKFold, train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import resample
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import xgboost as xgb

import pickle

matplotlib.rcParams['axes.labelsize'] = 14

class surrogate_model():
	def __init__(self, dataset):
		if not all([item in dataset.columns for item in ['Tk', 'Tg', 'n', 'Damage']]):
			raise ValueError("Input DataFrame have to contain columns named: Tk, Tg, n, damage")
		else:
			self.data = dataset


	def train(self, data_size_multipl_by = 3, split_val = 0.1, test_size = 0.2):
		n_samples = self.data.shape[0] * data_size_multipl_by
		# Define groups variable
		self.data['groups'] = [1 if f > split_val else 0 for f in self.data['Damage']]
		# Bootstraping by groups (equal sampling)
		data_resampled = resample(self.data, n_samples = n_samples, stratify = self.data['groups'])
		# Normalization
		data_features = data_resampled.loc[:, ['Tg', 'Tk', 'n']].values
		data_target = data_resampled['Damage'].values
		data_target = data_target.reshape(-1, 1)

		scaler_features, scaler_target = MinMaxScaler(), MinMaxScaler()
		scaler_features.fit(data_features)
		scaler_target.fit(data_target)

		self.scaler_features = scaler_features
		self.scaler_target = scaler_target

		data_features = scaler_features.transform(data_features)
		data_target = scaler_target.transform(data_target)

		X_train, X_test, y_train, y_test = train_test_split(data_features, 
															data_target, 
															test_size=test_size, 
															random_state=42)
		self.y_test = y_test

		# Train "weak" models
		model0 = lightGMB_model()
		model1_rs = adaboost_model()
		model2_rs = polynom_reg_model()
		model3_rs = catboost_model()

		kwargs = {'n_fold': 10, 'train': X_train, 'test': X_test, 'y': y_train}

		self.test_pred0, train_pred0, self.lightgbm_model = Stacking(model = model0, with_eval_set = True, **kwargs)
		self.test_pred1, train_pred1, self.adaboost_model = Stacking(model = model1_rs, **kwargs)
		self.test_pred2, train_pred2, self.polynom_reg_model = Stacking(model = model2_rs, **kwargs)
		self.test_pred3, train_pred3, self.catboost_model = Stacking(model = model3_rs, with_eval_set = True, **kwargs)

		# Train "strong" model over "weak" models predictions
		train_preds = pd.concat([train_pred0, train_pred1, train_pred2, train_pred3], axis = 1)
		test_preds = pd.concat([self.test_pred0, self.test_pred1, self.test_pred2, self.test_pred3], axis = 1)
		train_preds.columns = ['lightgbm_model', 'adaboost', 'polynom_reg', 'catboost']
		test_preds.columns = ['lightgbm_model', 'adaboost', 'polynom_reg', 'catboost']

		strong_model = xgboost_model(eval_set = [(X_test, y_test)])
		strong_model.fit(train_preds, y_train)
		self.strong_model = strong_model

		self.y_pred_test = strong_model.predict(test_preds)


	def print_test_metrics(self):
		print(f"RMSE score for LightGBM model: {sqrt(mean_squared_error(self.test_pred0, self.y_test))}")
		print(f"RMSE score for AdaBoost model: {sqrt(mean_squared_error(self.test_pred1, self.y_test))}")
		print(f"RMSE score for polynimial regression: {sqrt(mean_squared_error(self.test_pred2, self.y_test))}")
		print(f"RMSE score for CatBoost model: {sqrt(mean_squared_error(self.test_pred3, self.y_test))}")
		print(f"RMSE score for Stacked strong model: {sqrt(mean_squared_error(self.y_pred_test, self.y_test))}")

	def plot_test_true_vs_pred(self, figsize = (5, 5), title_text = "Stacked model", **kwargs):
		plt.figure(figsize = figsize)
		plot_true_vs_pred(self.y_pred_test, self.y_test.flatten(), title_text)

	def plot_test_true_minus_pred(self, figsize = (5, 5), title_text = "Stacked model", **kwargs):
		plt.figure(figsize = figsize)
		plot_true_minus_pred(self.y_pred_test, self.y_test.flatten(), title_text)

	def plot_submodels_test_true_vs_pred(self, figsize = (10, 10), **kwargs):
		fig, axes = plt.subplots(2, 2, figsize=figsize)
		axes = axes.flatten()

		plot_true_vs_pred(self.test_pred0.values.flatten(), self.y_test.flatten(), 'LightGBM model', ax=axes[0])
		plot_true_vs_pred(self.test_pred1.values.flatten(), self.y_test.flatten(), 'AdaBoost model', ax=axes[1])
		plot_true_vs_pred(self.test_pred2.values.flatten(), self.y_test.flatten(), 'Polynom_reg model', ax=axes[2])
		plot_true_vs_pred(self.test_pred3.values.flatten(), self.y_test.flatten(), 'CatBoost model', ax=axes[3])
		    
		fig.tight_layout()
		fig.show()


	def predict(self):
		# Normalize
		data_features = self.data.loc[:, ['Tg', 'Tk', 'n']]
		data_features = self.scaler_features.transform(data_features)
		# Make inference
		model1_preds = self.lightgbm_model.predict(data_features)
		model2_preds = self.adaboost_model.predict(data_features)
		model3_preds = self.polynom_reg_model.predict(data_features)
		model4_preds = self.catboost_model.predict(data_features)

		models_preds = pd.concat([pd.DataFrame(model1_preds), pd.DataFrame(model2_preds), 
		                          pd.DataFrame(model3_preds), pd.DataFrame(model4_preds)], axis = 1)
		models_preds.columns = ['lightgbm_model', 'adaboost', 'polynom_reg', 'catboost']

		damage_pred_scaled = self.strong_model.predict(models_preds)
		damage_pred_scaled = damage_pred_scaled.reshape(-1, 1)
		damage_pred = self.scaler_target.inverse_transform(damage_pred_scaled)

		return damage_pred, damage_pred_scaled

def plot_true_vs_pred(y_pred, y_test, title_text, ax=None, text_x_pos=0.17, text_y_pos=0.8, **kwargs):
    ax = ax or plt.gca()

    ax.set_title(title_text, y=1.01, fontsize=19, fontweight='bold')
    sns.scatterplot(x = y_pred, y = y_test, ax=ax, color='red', s=100)
    ax.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), color='black', linestyle='dashed')
    
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    ax.text(text_x_pos, text_y_pos, f'$RMSE={round(rmse, 4)}$', fontsize=12, fontweight='bold')
    ax.set_ylabel('$Damage_{true}$', fontsize = 15)
    ax.set_xlabel('$Damage_{pred}$', fontsize = 15)
    ax.grid()


def plot_true_minus_pred(y_pred, y_test, title_text, ax=None, text_x_pos=-0.015, text_y_pos=1350, **kwargs):
    ax = ax or plt.gca()
    
    ax.set_title(title_text, y=1.01, fontsize = 20, fontweight='bold')
    sns.distplot((y_test - y_pred), kde=True, norm_hist=True, color='red')

    rmse = sqrt(mean_squared_error(y_test, y_pred))
    ax.text(text_x_pos, text_y_pos, f'$RMSE={round(rmse, 4)}$', fontsize=12, fontweight='bold')

    ax.set_ylabel('$Occurence$', fontsize = 15)
    ax.set_xlabel('$Damage_{true}$ - $Damage_{pred}$', fontsize = 15)
    ax.grid()


# Define function for CV for "weak" models
def Stacking(model, train, y, test, n_fold, with_eval_set = False):
    folds = KFold(n_splits = n_fold, random_state = 42)
    test_pred = np.empty((0, 1), float)
    train_pred = np.empty((0, 1), float)
    y = y.flatten()
    for train_indices, val_indices in folds.split(train, y):
        x_train, x_val = train[train_indices], train[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        if with_eval_set == False:
            model.fit(x_train, y_train)
        else:
            model.fit(X = x_train, y = y_train, eval_set = [(x_val, y_val)], verbose = 0)
            
        train_pred = np.append(train_pred, model.predict(x_val))
        
    test_pred = np.append(test_pred, model.predict(test))
    return pd.DataFrame(test_pred), pd.DataFrame(train_pred), model

def lightGMB_model():
	# Define lightGBM model
	model0 = LGBMRegressor(boosting_type = 'gbdt',
	                       metric = 'rmse',
	                       n_estimators = 1000,
	                       num_boost_round = 1000,
	                       early_stopping_rounds = 100,
	                       # device = 'gpu',
	                       n_jobs = -1,
	                       random_state = 42,
	                       verbose = -1,
	                       verbose_eval = False)

	# Create the random grid
	model0_param_grid = {'learning_rate': [x for x in np.linspace(0.001, 0.05, num = 5)],
	                     'max_depth': [int(x) for x in np.linspace(2, 8, num = 4)]}

	model0_rs = RandomizedSearchCV(estimator = model0,
	                               param_distributions = model0_param_grid,
	                               n_iter=10, cv=3, verbose=0)
	return model0


def adaboost_model():
	# Define AdaBoost model
	model1 = AdaBoostRegressor(base_estimator = RandomForestRegressor(),
	                           loss = 'square',
	                           n_estimators = 1000,
	                           random_state = 42)

	# Create the random grid
	model1_param_grid = {'learning_rate': [x for x in np.linspace(0.001, 0.05, num = 5)],
	                     'base_estimator__max_depth': [int(x) for x in np.linspace(2, 8, num = 4)]}

	model1_rs = RandomizedSearchCV(estimator = model1,
	                               param_distributions = model1_param_grid,
	                               n_iter=10, cv=3, verbose=0)
	return model1_rs


def polynom_reg_model():
	# Define polynomial regression model
	model2 = make_pipeline(PolynomialFeatures(),
	                       LinearRegression(n_jobs = -1))

	model2_param_grid = {'polynomialfeatures__degree': [2, 3, 4, 5]}

	model2_rs = RandomizedSearchCV(estimator = model2,
	                               param_distributions = model2_param_grid,
	                               n_iter=10, cv=3, verbose=0)
	return model2_rs


def catboost_model():
	# Define CatBoost model
	model3 = CatBoostRegressor(loss_function='RMSE',
	                           random_seed = 42,
	                           task_type = 'GPU',
	                           # boosting_type = 'Plain', # set if it exceeds RAM
	                           # params for early stopping:
	                           iterations = 1000,
	                           od_type = 'Iter',
	                           od_wait = 100,
	                           silent = True)

	# Create the random grid
	model3_param_grid = {'learning_rate': [x for x in np.linspace(0.001, 0.05, num = 5)],
	                     'depth': [int(x) for x in np.linspace(2, 8, num = 4)]}

	model3_rs = RandomizedSearchCV(estimator = model3,
	                               param_distributions = model3_param_grid,
	                               n_iter=10, cv=3, verbose=0)
	return model3_rs


def xgboost_model(eval_set):
	strong_model = xgb.XGBRegressor(eval_metric = "rmse",
	                                # min_child_weight=1.5,                                                                  
	                                # reg_alpha=0.75,
	                                # reg_lambda=0.45,
	                                objective='reg:squarederror',
	                                n_estimators = 1000,
	                                num_boost_round = 1000,
	                                early_stopping_rounds = 100,
	                                # max_depth = 2,
	                                # num_leaves = 4,
	                                eval_set = eval_set,
	                                kvargs = {'tree_method':'gpu_hist'}, # enable GPU
	                                seed = 42)

	# Create the random grid
	strong_model_param_grid = {'colsample_bytree': [0.4, 0.6, 0.8],
	                           'reg_lambda': [0.01, 0.5, 1],
	                           'learning_rate': [x for x in np.linspace(0.001, 0.05, num = 5)],
	                           'max_depth': [int(x) for x in np.linspace(2, 8, num = 4)],
	                           'num_leaves': [20, 40, 60, 80]}


	strong_model = RandomizedSearchCV(estimator = strong_model,
	                                  param_distributions = strong_model_param_grid,
	                                  n_iter=100, cv=3, verbose=0)
	return strong_model



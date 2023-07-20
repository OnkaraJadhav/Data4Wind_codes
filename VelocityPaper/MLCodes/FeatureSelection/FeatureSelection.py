'''
Feature selection:
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from sklearn import metrics
# import seaborn as sns
# import matplotlib.pyplot as plt

def standardize(X_train, X_test):
    sc = StandardScaler()
    # Scale train features
    X_train_std = sc.fit_transform(X_train)
    # Use same scaling for test features
    X_test_std = sc.transform(X_test)
    return X_train_std, X_test_std

# Training data
rans = pd.read_csv('D:/Data4wind/MLCourse/FeatureSelection/RANS-a0_a45_BB.txt', sep = ' ')

# Features
X = rans.values[:,1:13]

# Load LES data
les = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/LES/LES-a0_a45_BB.txt', sep = ' ')
    
# Labels
y = les.values[:,4:5] # Ux

y = y.ravel()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

X_train, X_test = standardize(X_train, X_test)

#%%
# Feature selection (F Statistics)
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
 
# feature selection
def select_features(X_train, y_train):
	# configure to select all features
	fs = SelectKBest(score_func=f_regression, k='all') #
	# learn relationship from training data
	fs.fit(X_train, y_train)
	return fs

# load the dataset
# feature selection
fs = select_features(X_train, y_train)
# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
#%%
# Feature selection (MI)
from sklearn.feature_selection import mutual_info_regression
 
# feature selection
def select_features(X_train, y_train):
	# configure to select all features
	fs = SelectKBest(score_func=mutual_info_regression, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	return fs

# load the dataset
# feature selection
fs = select_features(X_train, y_train)
# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))

#%% 
# Feature selection (recursive feature elimination)
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingRegressor

estimator  = GradientBoostingRegressor(n_estimators= 1750, max_depth=2, warm_start=True)

selector = RFE(estimator, n_features_to_select=8, step=1)
selector = selector.fit(X_train, y_train)

selector.ranking_

#%% 
# Feature selection (Sequential Feature Selector)

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import GradientBoostingRegressor

estimator  = GradientBoostingRegressor(n_estimators= 1750, max_depth=2, warm_start=True)

selector = SequentialFeatureSelector(estimator, n_features_to_select=8, direction='backward')
selector = selector.fit(X_train, y_train)

selector.ranking_

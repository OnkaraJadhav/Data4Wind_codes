'''
Author: Onkar Jadhav
Python script for hyperparameter optimization of RandomForestRegressor
'''

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd
from sklearn import metrics

from sklearn.preprocessing import StandardScaler

def standardize(X_train, X_test_ang):
    sc = StandardScaler()
    # Scale train features
    X_train_std = sc.fit_transform(X_train)
    # Use same scaling for test features
    X_test_ang_std = sc.transform(X_test_ang)
    return X_train_std, X_test_ang_std

# Training data
rans = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/RANS/RANS-a0_a45_BB.txt', sep = ' ')

# Testing data for different angle
rans_ang = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/SB/RANS-a15_SB.txt', sep = ' ')
# Features
#X = rans.values[:,np.r_[1, 3, 4:10]]
#X_test_ang = rans_ang.values[:,np.r_[1, 3, 4:10]]

X = rans.values[:,1:13]
X_test_ang = rans_ang.values[:,1:13]

# Load LES data
les = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/LES/LES-a0_a45_BB.txt', sep = ' ')
    
# Labels (Output)
y = les.values[:,10:11] # 
y = y.ravel()

# Standardize the data
X, X_test_ang = standardize(X, X_test_ang)

# hyperparameter space
p_space = {'n_estimators':np.linspace(100,2000, num=10, dtype = 'int'), 'max_depth': np.linspace(10, 110, num = 11), 
           'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'bootstrap': [True, False] }

# randomized search for hyperparameter optimization
tun_model = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions = p_space,
                               scoring = 'r2', n_iter=50, n_jobs=-1, cv=3)

tun_model.fit(X,y)

# Get the best hyperparameters
# I have used these hyperparameters in 'Optimized_RFRegression.py'
tun_model.best_params_, tun_model.best_score_

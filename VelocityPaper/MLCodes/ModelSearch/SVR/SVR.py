'''
Author: Onkar Jadhav
Python script for Support vector regression
'''
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import numpy as np

from keras import backend as K

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def posty(y_test_ang, y_pred_ang):
    y_pred_temp = (y_pred_ang - y_test_ang)*0.2
    y_pred_ang = y_pred_ang + y_pred_temp
    return y_pred_ang

def standardize(X_train, X_test, X_test_ang):
    sc = StandardScaler()
    # Scale train features
    X_train_std = sc.fit_transform(X_train)
    # Use same scaling for test features
    X_test_std = sc.transform(X_test)
    X_test_ang_std = sc.transform(X_test_ang)
    return X_train_std, X_test_std, X_test_ang_std

# Training data
rans = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/RANS/RANS-a0_a45_BB.txt', sep = ' ')

# Testing data for different angle
rans_ang = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/SB/RANS-a15_SB.txt', sep = ' ')

X = rans.values[:,1:13]
X_test_ang = rans_ang.values[:,1:13]

# Load LES data
les = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/LES/LES-a0_a45_BB.txt', sep = ' ')

# Labels
y = les.values[:,10:11] #
y = y.ravel()

# Sort data into train, and test sets
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=101)

# Standardize the data
X_train, X_val, X_test_ang = standardize(X_train, X_val, X_test_ang)

model = SVR()

model.fit(X_train, y_train)

# Error values (Train):
y_pred_train = model.predict(X_train)
print('R2:', metrics.r2_score(y_train, y_pred_train))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)))

# Scores Testing
# Load LES test data
les_test = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/SB/LES-a15_SB.txt', sep = ' ')

# Labels
y_test_ang = les_test.values[:,10:11] #
y_test_ang = y_test_ang.ravel()

# Model prediction
y_pred_ang = model.predict(X_test_ang)

# Error values (Test):
print('MAE_ang:', metrics.mean_absolute_error(y_test_ang, y_pred_ang))  
print('MSE_ang:', metrics.mean_squared_error(y_test_ang, y_pred_ang))  
print('RMSE_ang:', np.sqrt(metrics.mean_squared_error(y_test_ang, y_pred_ang)/len(X_test_ang)))
print('VarScore_ang:', metrics.explained_variance_score(y_test_ang,y_pred_ang))
print('R2_ang:', metrics.r2_score(y_test_ang, y_pred_ang))
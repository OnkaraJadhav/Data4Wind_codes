# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 11:20:45 2022

@author: onkar.jadhav
"""

# Import necessary files
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Import the cluster models:
from ModelPos1 import *
from ModelPos2 import *
from ModelNeg1 import *
from ModelNeg2 import *

# Import training dataset:
rans = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset_1B/RANS/RANS-a0_a45_OB.txt', sep = ' ')

les = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset_1B/LES/LES-a0_a45_OB.txt', sep = ' ')

# Testing data for different angle
rans_ang = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/SB/RANS-a15_SB.txt', sep = ' ')
    
les_test = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/SB/LES-a15_SB.txt', sep = ' ')
    
X = rans.values[:,0:13] 
X_test_ang = rans_ang.values[:,0:13]
    
y_test_ang = les_test.values[:,6:7]
y = les.values[:,6:7]

# Concatenate the data to keep the indices
X_y = np.concatenate((X, y), axis = 1)

# sort the data into negative and positive part
X_y_neg = X_y[X_y[:,12] < 0, :] # negative
X_y_pos = X_y[X_y[:,12] >= 0, :] # positive

#%% Clusters for negative part

kmeans_neg = KMeans(n_clusters=2, random_state=0).fit(X_y_neg[:,12].reshape(-1,1))
KLabels_neg = kmeans_neg.labels_

KLabels_neg = np.resize(KLabels_neg, (len(KLabels_neg),1))
X_y_clust_neg = np.concatenate((X_y_neg, KLabels_neg), axis = 1)

X_y_c1_neg = X_y_clust_neg[X_y_clust_neg[:,13] == 0, :]
X_y_c2_neg = X_y_clust_neg[X_y_clust_neg[:,13] == 1, :]
# X_y_c3_neg = X_y_clust_neg[X_y_clust_neg[:,13] == 2, :]

# Training set for cluster 1
X_c1_neg = X_y_c1_neg[:,1:12]
y_c1_neg = X_y_c1_neg[:,12]
y_c1_neg = np.resize(y_c1_neg, (len(y_c1_neg),1))

# Training set for cluster 2
X_c2_neg = X_y_c2_neg[:,1:12]
y_c2_neg = X_y_c2_neg[:,12]
y_c2_neg = np.resize(y_c2_neg, (len(y_c2_neg),1))

# # Training set for cluster 3
# X_c3_neg = X_y_c3_neg[:,1:12]
# y_c3_neg = X_y_c3_neg[:,12]
# y_c3_neg = np.resize(y_c3_neg, (len(y_c3_neg),1))

#%% Clsters for positive part

kmeans_pos = KMeans(n_clusters=2, random_state=0).fit(X_y_pos[:,12].reshape(-1,1))
KLabels_pos = kmeans_pos.labels_
KLabels_pos = np.resize(KLabels_pos, (len(KLabels_pos),1))

X_y_clust_pos = np.concatenate((X_y_pos, KLabels_pos), axis = 1)

X_y_c1_pos = X_y_clust_pos[X_y_clust_pos[:,13] == 0, :]
X_y_c2_pos = X_y_clust_pos[X_y_clust_pos[:,13] == 1, :]
# X_y_c3_pos = X_y_clust_pos[X_y_clust_pos[:,13] == 2, :]

# Training set for cluster 1
X_c1_pos = X_y_c1_pos[:,1:12]
y_c1_pos = X_y_c1_pos[:,12]
y_c1_pos = np.resize(y_c1_pos, (len(y_c1_pos),1))

# Training set for cluster 2
X_c2_pos = X_y_c2_pos[:,1:12]
y_c2_pos = X_y_c2_pos[:,12]
y_c2_pos = np.resize(y_c2_pos, (len(y_c2_pos),1))

# # Training set for cluster 3
# X_c3_pos = X_y_c3_pos[:,1:12]
# y_c3_pos = X_y_c3_pos[:,12]
# y_c3_pos = np.resize(y_c3_pos, (len(y_c3_pos),1))

#%% Same for test - Positive and negative part

les_test_OB = pd.read_csv('D:/Data4wind/VelocityStudy_v1/ANN/15/UzSplitTest/Uz_OB.csv', sep = ',')
y_test_ang_OB = les_test_OB.values[:,4].reshape(-1,1)

# Concatenate the data to keep the indices
X_y_test = np.concatenate((X_test_ang, y_test_ang_OB, y_test_ang), axis = 1)

# sort the data into negative and positive part
X_y_neg_test = X_y_test[X_y_test[:,12] < 0, :] # negative
X_y_pos_test = X_y_test[X_y_test[:,12] >= 0, :] # positive

#%% Clusters for negative part Test

KLabels_test_neg = kmeans_neg.predict(X_y_neg_test[:,12].reshape(-1, 1))
KLabels_test_neg = np.resize(KLabels_test_neg, (len(KLabels_test_neg),1))

X_y_clust_test_neg = np.concatenate((X_y_neg_test, KLabels_test_neg), axis = 1)

X_y_c1_neg_test = X_y_clust_test_neg[X_y_clust_test_neg[:,14] == 0, :]
X_y_c2_neg_test = X_y_clust_test_neg[X_y_clust_test_neg[:,14] == 1, :]
# X_y_c3_neg_test = X_y_clust_test_neg[X_y_clust_test_neg[:,14] == 2, :]

# Testing set for cluster 1
X_c1_neg_test = X_y_c1_neg_test[:,1:12]
y_c1_neg_test = X_y_c1_neg_test[:,13]
y_c1_neg_test = np.resize(y_c1_neg_test, (len(y_c1_neg_test),1))

# Testing set for cluster 2
X_c2_neg_test = X_y_c2_neg_test[:,1:12]
y_c2_neg_test = X_y_c2_neg_test[:,13]
y_c2_neg_test = np.resize(y_c2_neg_test, (len(y_c2_neg_test),1))

# # Testing set for cluster 3
# X_c3_neg_test = X_y_c3_neg_test[:,1:12]
# y_c3_neg_test = X_y_c3_neg_test[:,13]
# y_c3_neg_test = np.resize(y_c3_neg_test, (len(y_c3_neg_test),1))

#%% Clusters for positive part Test

KLabels_test_pos = kmeans_pos.predict(X_y_pos_test[:,12].reshape(-1, 1))
KLabels_test_pos = np.resize(KLabels_test_pos, (len(KLabels_test_pos),1))

X_y_clust_test_pos = np.concatenate((X_y_pos_test, KLabels_test_pos), axis = 1)

X_y_c1_pos_test = X_y_clust_test_pos[X_y_clust_test_pos[:,14] == 0, :]
X_y_c2_pos_test = X_y_clust_test_pos[X_y_clust_test_pos[:,14] == 1, :]
# X_y_c3_pos_test = X_y_clust_test_pos[X_y_clust_test_pos[:,14] == 2, :]

# Testing set for cluster 1
X_c1_pos_test = X_y_c1_pos_test[:,1:12]
y_c1_pos_test = X_y_c1_pos_test[:,13]
y_c1_pos_test = np.resize(y_c1_pos_test, (len(y_c1_pos_test),1))

# Testing set for cluster 2
X_c2_pos_test = X_y_c2_pos_test[:,1:12]
y_c2_pos_test = X_y_c2_pos_test[:,13]
y_c2_pos_test = np.resize(y_c2_pos_test, (len(y_c2_pos_test),1))

# # Testing set for cluster 2
# X_c3_pos_test = X_y_c3_pos_test[:,1:12]
# y_c3_pos_test = X_y_c3_pos_test[:,13]
# y_c3_pos_test = np.resize(y_c3_pos_test, (len(y_c3_pos_test),1))

#%%
# Positive
RMSE_pos_c1, R2_pos_c1, y_pred_pos_c1 = modelpos1(X_c1_pos, X_c1_pos_test, y_c1_pos, y_c1_pos_test)

RMSE_pos_c2, R2_pos_c2, y_pred_pos_c2 = modelpos2(X_c2_pos, X_c2_pos_test, y_c2_pos, y_c2_pos_test)

# RMSE_pos_c3, R2_pos_c3, y_pred_pos_c3 = modelpos1(X_c3_pos, X_c3_pos_test, y_c3_pos, y_c3_pos_test)

# Negative
RMSE_neg_c1, R2_neg_c1, y_pred_neg_c1 = modelneg1(X_c1_neg, X_c1_neg_test, y_c1_neg, y_c1_neg_test)

RMSE_neg_c2, R2_neg_c2, y_pred_neg_c2 = modelneg2(X_c2_neg, X_c2_neg_test, y_c2_neg, y_c2_neg_test)

# RMSE_neg_c3, R2_neg_c3, y_pred_neg_c3 = modelneg1(X_c3_neg, X_c3_neg_test, y_c3_neg, y_c3_neg_test)

y_pred = np.concatenate((y_pred_neg_c1, y_pred_neg_c2, y_pred_pos_c1, y_pred_pos_c2), axis = 0) #, y_pred5
y_test_new = np.concatenate((y_c1_neg_test, y_c2_neg_test, y_c1_pos_test, y_c2_pos_test), axis = 0)

print('MAE_ang:', metrics.mean_absolute_error(y_test_new, y_pred))  
print('MSE_ang:', metrics.mean_squared_error(y_test_new, y_pred))  
print('RMSE_ang:', np.sqrt(metrics.mean_squared_error(y_test_new, y_pred)))
print('variance:', metrics.explained_variance_score(y_test_new,y_pred))
print('R2_ang:', metrics.r2_score(y_test_new, y_pred))
#%%
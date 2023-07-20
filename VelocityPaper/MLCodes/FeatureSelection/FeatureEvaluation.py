# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 11:20:45 2022

@author: onkar.jadhav
"""


import numpy as np
import pandas as pd

from EvlFunction import *

Temp = [9,11,2,4,6,1,8,3,5]

iters = 3

RMSEs = []
R2s=[]

RMSE1s = []
R21s=[]

RMSE2s = []
R22s=[]

RMSE3s = []
R23s=[]


les_test = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/SB/LES-a15_SB.txt', sep = ' ')
    
# Labels
y_test_ang = les_test.values[:,6:7]
y_test_ang = y_test_ang

y_pred_angs = np.zeros([len(y_test_ang),3])

for x in range(iters):
    
    rans = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset_1B/RANS/RANS-a0_a45_OB.txt', sep = ' ')

    # Testing data for different angle
    rans_ang = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/SB/RANS-a15_SB.txt', sep = ' ')
    
    les = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset_1B/LES/LES-a0_a45_OB.txt', sep = ' ')
    
    les_test = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/SB/LES-a15_SB.txt', sep = ' ')
    
    X = rans.values[:,Temp] 
    X_test_ang = rans_ang.values[:,Temp]
    
    del Temp[-1]
    
    y_test_ang = les_test.values[:,6:7]
    y = les.values[:,6:7]
    
    RMSE, R2, y_pred_ang = main(X, X_test_ang, y, y_test_ang) #, RMSE1, R21, RMSE2, R22, RMSE3, R23
    
    RMSEs = np.append(RMSEs, RMSE)
    R2s = np.append(R2s, R2)
    
    y_pred_angs = np.append(y_pred_angs, y_pred_ang, axis=1)
    print(x)
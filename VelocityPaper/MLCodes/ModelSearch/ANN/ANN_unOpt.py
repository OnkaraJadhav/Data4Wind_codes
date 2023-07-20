'''
Author: Onkar Jadhav
Python script for a 4 layer artificial neural network
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout
from keras.callbacks import ReduceLROnPlateau

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from keras import backend as K

'''
import xlsxwriter
def write_excel(numpy_array):
    workbook = xlsxwriter.Workbook('D:/Data4wind/Results_aft_Rotation/ModelSearch/Ann-0_45/15/ML-a15_temp_6f_mcp.xlsx')
    worksheet = workbook.add_worksheet()

    for row, data in enumerate(numpy_array):
        worksheet.write_row(row, 0, data)
        #worksheet.write_number(0, row, data)
    workbook.close()
'''

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

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
rans_ang = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/SB/RANS-a30_SB.txt', sep = ' ')
#%%
X = rans.values[:,1:13]
X_test_ang = rans_ang.values[:,1:13]

# Load LES data
les = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/LES/LES-a0_a45_BB.txt', sep = ' ')
       
#les_a = pd.read_csv('C:/Users/rpolz/Desktop/Onkar/LuxWork/Data4Wind/Dataset/LES/LES-a45.txt', sep = ' ')

# Labels
y = les.values[:,4:7] # 
# y = y.ravel()
#%%
# Sort data into train, and test sets
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=101)

# Standardize the data
X_train, X_val, X_test_ang = standardize(X_train, X_val, X_test_ang)

# Define layers 
X_rows, input_shape = np.shape(X)
inputs = keras.Input(shape=(input_shape,)) # 
x = layers.Dense(10, activation='linear')(inputs)  
x = layers.Dense(10, activation='relu')(x) # 
# x = layers.Dense(10, activation='relu')(x) # 
x = layers.Dense(10, activation='relu')(x) # 
x = layers.Dense(10, activation='relu')(x) # 
outputs = layers.Dense(3, activation='linear')(x)

# Model:
model = keras.Model(inputs,outputs, name='model')

# Optimzer: Here used (Adam)
opt = keras.optimizers.Adam(learning_rate=0.01) #Adam, SGD, RMSprop, Nadam, Adamax

# Compile model with loss function as mean squared error
# metrics determines the training root mean square error as well as R2 values.
# The last epoch values are the final RMSE and R2.
model.compile(optimizer=opt,
              loss='mse',
              metrics=[tf.keras.metrics.RootMeanSquaredError(), coeff_determination]
              )

# Fit model for train data and validate it on validation data
model.fit(x=X_train,y=y_train,
          validation_data=(X_val,y_val),
          batch_size=64,epochs=30)
model.summary()
#%% Testing
# Save model

# Scores Testing
# Load LES test data
les_test = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/SB/LES-a30_SB.txt', sep = ' ')

# Labels
#y_test_ang = les_test.values[:,4:5] # CpMean
y_test_ang1 = les_test.values[:,4:7] # CpPrime
# y_test_ang = y_test_ang.ravel()

# Model prediction
y_pred_ang1 = model.predict(X_test_ang)


#write_excel(y_pred_ang)
#%%
y_pred_ang = y_pred_ang1[:,0:1]
y_test_ang = y_test_ang1[:,0:1]
# Error values (Wind tunnel):
print('MAE_ang:', metrics.mean_absolute_error(y_test_ang, y_pred_ang))  
print('MSE_ang:', metrics.mean_squared_error(y_test_ang, y_pred_ang))  
print('RMSE_ang:', np.sqrt(metrics.mean_squared_error(y_test_ang, y_pred_ang)))
print('VarScore_ang:', metrics.explained_variance_score(y_test_ang,y_pred_ang))
print('R2_ang:', metrics.r2_score(y_test_ang, y_pred_ang))
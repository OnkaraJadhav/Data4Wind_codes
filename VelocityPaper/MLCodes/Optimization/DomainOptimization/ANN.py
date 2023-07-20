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
from tensorflow.keras import initializers
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from keras import backend as K
#%%
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )



def standardize(X_train, X_test, X_test_ang):
    sc = MinMaxScaler()
    # Scale train features
    X_train_std = sc.fit_transform(X_train)
    # Use same scaling for test features
    X_test_std = sc.transform(X_test)
    X_test_ang_std = sc.transform(X_test_ang)
    return X_train_std, X_test_std, X_test_ang_std

'''
import xlsxwriter
def write_excel(numpy_array):
    workbook = xlsxwriter.Workbook('D:/Data4wind/Results_aft_symmetric/0-225-45/15/ML-a15_temp_6f.xlsx')
    worksheet = workbook.add_worksheet()

    for row, data in enumerate(numpy_array):
        worksheet.write_row(row, 0, data)
        #worksheet.write_number(0, row, data)
    workbook.close()
'''
def fity(y_pred_ang, y_test_ang):
    y_pred_temp = (y_pred_ang - y_test_ang)*0.2
    y_pred_ang = y_pred_ang - y_pred_temp
    return y_pred_ang

def Norml(X_train, X_test, X_test_ang):
    sc = StandardScaler()
    # Scale train features
    X_train_std = sc.fit_transform(X_train)
    # Use same scaling for test features
    X_test_std = sc.transform(X_test)
    X_test_ang_std = sc.transform(X_test_ang)
    return X_train_std, X_test_std, X_test_ang_std
#%%
# Training data
# rans = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset_15B/RANS/RANS-a0_a45_a225_BB.txt', sep = ' ')
#rans = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/SB/RANS-a0_a45_a225_SB.txt', sep = ' ')
# rans = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/RANS/RANS-a0_a45_BB.txt', sep = ' ')
rans = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset_1B/RANS/RANS-a0_a225_a45_OB.txt', sep = ' ')


# Testing data for different angle
rans_ang = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/SB/RANS-a15_SB.txt', sep = ' ')
# rans_ang = pd.read_csv('D:/Data4wind/VelocityStudy_v1/DATASET/SmallBox_test/RANS-a15_SB.txt', sep = ' ')

# Features
#X = rans.values[:,np.r_[1, 3, 4, 7, 8, 9]] #1, 3, 4:6, 7, 8, 9
#X_test_ang = rans_ang.values[:,np.r_[1, 3, 4, 7, 8, 9]] #1, 3, 4:6, 7, 8, 9

X = rans.values[:,1:13]
X_test_ang = rans_ang.values[:,1:13]

# Load LES data
# les = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset_15B/LES/LES-a0_a45_a225_BB.txt', sep = ' ')
#les = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/SB/LES-a0_a45_a225_SB.txt', sep = ' ')
# les = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/LES/LES-a0_a45_BB.txt', sep = ' ')
les = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset_1B/LES/LES-a0_a225_a45_OB.txt', sep = ' ')
#%%
#les_a = pd.read_csv('C:/Users/rpolz/Desktop/Onkar/LuxWork/Data4Wind/Dataset/LES/LES-a45.txt', sep = ' ')

# Labels
y = les.values[:,6:7] # U_x
#%%
#y = les.values[:,4:5] # CpMean
# y = y.ravel()
#scaler_out = MinMaxScaler()
# fit scaler on data
#scaler_out.fit(y)
#y = scaler_out.transform(y)
# Load LES test data
#les_x = pd.read_csv('D:/Data4wind/Results_aft_Rotation/LES_Rotated/LES-a30.txt', sep = ' ')

# Labels
#y_x_ang = les_x.values[:,4:5] # CpMean
#y_x_ang = les_x.values[:,5:6] # CpPrime
#y_x_ang = y_x_ang.ravel()
start = time.time()

from sklearn.model_selection import train_test_split

#X_temp, X_add, y_temp, y_add = train_test_split(X_test_ang, les_add, test_size=0.001, random_state=101)

#X_t, X_val, y_t, y_val = train_test_split(X_test_ang, y_x_ang, test_size=0.7, random_state=101)

# Sort data into train, and test sets

#X_temp, X_add, y_temp, y_add = train_test_split(X_test_ang, les_add, test_size=0.001, random_state=101)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=101)

#X_train = np.concatenate([X_train, X_add])
#y_train = np.concatenate([y_train, y_add])
"""
y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)
TargetScaler = MinMaxScaler()
TargetScaler.fit_transform(y_train)
y_train = TargetScaler.transform(y_train)
y_val = TargetScaler.transform(y_val)
"""

# Standardize the data
X_train, X_val, X_test_ang = standardize(X_train, X_val, X_test_ang)

## Neural Network Model
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.00001)

"""
# Define layers
X_rows, input_shape = np.shape(X)
inputs = keras.Input(shape=(input_shape,)) # 
x = layers.Dense(64, activation='relu')(inputs)  
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)
#x = Dropout(0.2)(x)
#x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(1, activation='linear')(x) #kernel_regularizer=keras.regularizers.l2(l=0.001)
"""


######################CpMean################
initializer =  tf.keras.initializers.glorot_uniform()

X_rows, input_shape = np.shape(X)
inputs = keras.Input(shape=(input_shape,)) # 
x = layers.Dense(64, activation='tanh', kernel_initializer=initializer)(inputs)  
x = layers.Dense(128, activation='tanh', kernel_initializer=initializer)(x) # Use 'relu' activation for CpPrime
x = Dropout(0.25)(x)
x = layers.Dense(128, activation='tanh', kernel_initializer=initializer)(x)
x = layers.Dense(64, activation='tanh', kernel_initializer=initializer)(x)
x = layers.Dense(64, activation='tanh', kernel_initializer=initializer)(x)
outputs = layers.Dense(1, activation='linear')(x)

"""
initializer =  tf.keras.initializers.he_uniform()

X_rows, input_shape = np.shape(X)
inputs = keras.Input(shape=(input_shape,)) # 
x = layers.Dense(64, activation='linear', kernel_initializer=initializer)(inputs) 
x = layers.LeakyReLU(alpha=0.3)(x) # Use 'relu' activation for CpPrime
x = layers.Dense(128, activation='relu', kernel_initializer=initializer)(x)
x = layers.LeakyReLU(alpha=0.3)(x) # Use 'relu' activation for CpPrime
x = Dropout(0.3)(x)
x = layers.Dense(128, activation='relu', kernel_initializer=initializer)(x)
x = layers.LeakyReLU(alpha=0.3)(x) # Use 'relu' activation for CpPrime
#x = Dropout(0.15)(x)
x = layers.Dense(128, activation='relu', kernel_initializer=initializer)(x)
x = layers.LeakyReLU(alpha=0.3)(x) # Use 'relu' activation for CpPrime
#x = Dropout(0.3)(x)
x = layers.Dense(64, activation='linear', kernel_initializer=initializer)(x)
x = layers.LeakyReLU(alpha=0.3)(x) # Use 'relu' activation for CpPrime
outputs = layers.Dense(1, activation='linear')(x)
"""

"""
initializer =  tf.keras.initializers.he_normal()

X_rows, input_shape = np.shape(X)
inputs = keras.Input(shape=(input_shape,)) # 
x = layers.Dense(64, activation='relu', kernel_initializer=initializer)(inputs) 
# x = layers.Dense(64, activation='relu', kernel_initializer=initializer)(x)
x = layers.Dense(128, activation='relu', kernel_initializer=initializer)(x) # Use 'relu' activation for CpPrime
x = Dropout(0.2)(x)
x = layers.Dense(128, activation='relu', kernel_initializer=initializer)(x)
x = Dropout(0.2)(x)
x = layers.Dense(64, activation='relu', kernel_initializer=initializer)(x)
x = layers.Dense(64, activation='relu', kernel_initializer=initializer)(x)
outputs = layers.Dense(4, activation='linear')(x)
"""

"""
initializer =  tf.keras.initializers.he_normal()

X_rows, input_shape = np.shape(X)
inputs = keras.Input(shape=(input_shape,)) # 
x = layers.Dense(160, activation='relu', kernel_initializer=initializer)(inputs) 
#x = layers.Dense(160, activation='relu', kernel_initializer=initializer)(x)
#x = layers.LeakyReLU(alpha=0.3)(x) # Use 'relu' activation for CpPrime
x = Dropout(0.1)(x)
x = layers.Dense(96, activation='relu', kernel_initializer=initializer)(x)
#x = layers.Dense(96, activation='relu', kernel_initializer=initializer)(x)
#x = layers.LeakyReLU(alpha=0.3)(x) # Use 'relu' activation for CpPrime
#x = Dropout(0.15)(x)
x = layers.Dense(192, activation='relu', kernel_initializer=initializer)(x)
#x = layers.LeakyReLU(alpha=0.3)(x) # Use 'relu' activation for CpPrime
x = layers.Dense(176, activation='relu', kernel_initializer=initializer)(x)
outputs = layers.Dense(4, activation='linear')(x)
"""


# Model:
model = keras.Model(inputs,outputs, name='model')


from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#%%
# Optimzer: Here used (Root Mean Squared Propagation)
opt = keras.optimizers.Adamax(learning_rate=0.001)

# Compile model with loss function as mean squared error
# metrics determines the training root mean square error as well as R2 values.
# The last epoch values are the final RMSE and R2.
model.compile(optimizer=opt,
              loss='mse',
              metrics=[tf.keras.metrics.RootMeanSquaredError(), coeff_determination]
              )

#from sklearn.compose import TransformedTargetRegressor
#wrapped_model = TransformedTargetRegressor(regressor=model, transformer=MinMaxScaler())

# Fit model for train data and validate it on validation data
model.fit(X_train,y_train,
          validation_data=(X_val,y_val),
          batch_size=32,epochs=30, callbacks=[reduce_lr])
model.summary()

end = time.time()
total_time = end - start
print("\n"+ str(total_time))
# Save model

# Scores Testing
# Load LES test data
les_test = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/SB/LES-a15_SB.txt', sep = ' ')

# les_test = pd.read_csv('D:/Data4wind/VelocityStudy_v1/DATASET/SmallBox_test/LES-a15_SB.txt', sep = ' ')

# Labels
#y_test_ang = les_test.values[:,4:5] # CpMean
y_test_ang = les_test.values[:,6:7] # CpPrime
#y_test_ang = y_test_ang.ravel()
#y_test_ang = scaler_out.transform(y_test_ang)
# Model prediction
y_pred_ang = model.predict(X_test_ang)

# inverse transform
#y_pred_ang = scaler_out.inverse_transform(y_pred_ang)

#y_pred_ang = TargetScaler.inverse_transform(y_pred_ang)
#write_excel(y_pred_ang)

# Error values (Wind tunnel):
# y_pred_ang = fity(y_pred_ang, y_test_ang)
y_pred_ang1 = y_pred_ang[:,0:1]
y_test_ang1 = y_test_ang[:,0:1]
print('MAE_ang:', metrics.mean_absolute_error(y_test_ang1, y_pred_ang1))  
print('MSE_ang:', metrics.mean_squared_error(y_test_ang1, y_pred_ang1))  
print('RMSE_ang:', np.sqrt(metrics.mean_squared_error(y_test_ang1, y_pred_ang1)))
print('variance:', metrics.explained_variance_score(y_test_ang1,y_pred_ang1))
print('R2_ang:', metrics.r2_score(y_test_ang1, y_pred_ang1))


###########
"""
from sklearn.metrics import mean_absolute_percentage_error
nu_exp = np.array([76.79981973840628,126.72721291263258,166.81879205076086,202.38082395592144,234.0354303283891,264.0558029259876,292.8577903728362,320.13579935136903,345.3502847623827])

y = np.array([48.0449,109.8245,166.8188,217.6617,258.8140,290.5560,311.3228,320.1358,317.8285])

Err = mean_absolute_percentage_error(nu_exp,y)
"""
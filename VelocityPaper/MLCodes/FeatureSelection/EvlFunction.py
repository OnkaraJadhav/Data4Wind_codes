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

RMSE = []
R2 = []
y_pred_ang = []

RMSE1 = []
R21 = []

RMSE2 = []
R22 = []

RMSE3 = []
R23 = []


def main(X, X_test_ang, y, y_test_ang):
    global RMSE, R2,  y_pred_ang #RMSE1, R21, RMSE2, R22, RMSE3, R23
    start = time.time()
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=101)
    

    # Standardize the data
    X_train, X_val, X_test_ang = standardize(X_train, X_val, X_test_ang)

    ## Neural Network Model
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=2, min_lr=0.00001)

        
    initializer =  tf.keras.initializers.he_uniform()
    
    X_rows, input_shape = np.shape(X)
    inputs = keras.Input(shape=(input_shape,)) # 
    x = layers.Dense(64, activation='relu', kernel_initializer=initializer)(inputs) 
    #x = layers.Dense(64, activation='relu', kernel_initializer=initializer)(x)
    x = layers.Dense(128, activation='relu', kernel_initializer=initializer)(x) # Use 'relu' activation for CpPrime
    x = Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu', kernel_initializer=initializer)(x)
    x = Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu', kernel_initializer=initializer)(x)
    x = layers.Dense(64, activation='relu', kernel_initializer=initializer)(x)
    outputs = layers.Dense(1, activation='linear')(x)
    
    
    # Model:
    model = keras.Model(inputs,outputs, name='model')
    
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
    
    # Model prediction
    y_pred_ang = model.predict(X_test_ang)
    
    end = time.time()
    total_time = end - start
    print("\n"+ str(total_time))
    
    # Error values (Wind tunnel):
    y_pred_ang1 = y_pred_ang[:,0:1]
    y_test_ang1 = y_test_ang[:,0:1]

    RMSE = np.sqrt(metrics.mean_squared_error(y_test_ang1, y_pred_ang1))
    R2 = metrics.r2_score(y_test_ang1, y_pred_ang1)

    print('MAE_ang:', metrics.mean_absolute_error(y_test_ang1, y_pred_ang1))  
    print('MSE_ang:', metrics.mean_squared_error(y_test_ang1, y_pred_ang1))  
    print('RMSE_ang:', np.sqrt(metrics.mean_squared_error(y_test_ang1, y_pred_ang1)))
    print('variance:', metrics.explained_variance_score(y_test_ang1,y_pred_ang1))
    print('R2_ang:', metrics.r2_score(y_test_ang1, y_pred_ang1))
        
    return RMSE, R2, y_pred_ang #, RMSE1, R21, RMSE2, R22, RMSE3, R23
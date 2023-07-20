import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import keras_tuner as kt

from keras import backend as K

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
rans = pd.read_csv('D:/Data4wind/VelocityStudy_v1/DATASET/RANS_All/RANS-a0_a45_BB.txt', sep = ' ')

# Testing data for different angle
rans_ang = pd.read_csv('D:/Data4wind/VelocityStudy_v1/DATASET/RANS_SB/RANS-a15_SB.txt', sep = ' ')

# Features
X = rans.values[:,1:13]
X_test_ang = rans_ang.values[:,1:13]
X = np.concatenate((X,X_test_ang)) 
# Load LES data
les = pd.read_csv('D:/Data4wind/VelocityStudy_v1/DATASET/LES/LES-a0_a45_BB.txt', sep = ' ')
les_test = pd.read_csv('D:/Data4wind/VelocityStudy_v1/DATASET/SB/LES-a15_SB.txt', sep = ' ')

# Labels
#y_test_ang = les_test.values[:,4:5] # CpMean
y_test_ang = les_test.values[:,10:11] # CpPrime
# Labels
y = les.values[:,10:11] # U_x
#y = les.values[:,4:5] # CpMean
y = np.concatenate((y,y_test_ang))

# Sort data into train, and test sets
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=101)

# Standardize the data
X_train, X_val, X_test_ang = standardize(X_train, X_val, X_test_ang)

X_rows, Input_shape = np.shape(X) # shape of an input

#class MyHyperModel(kt.HyperModel):
def build_model(hp):
    """
    Builds model and sets up hyperparameter space to search.
        
    Parameters
        ----------
        hp : HyperParameter object
        Configures hyperparameters to tune.
            
        Returns
            -------
            model : keras model
            Compiled model with hyperparameters to tune.
    """
    # Initialize sequential API and start building model.
    model = keras.Sequential()
    model.add(keras.Input(shape=(Input_shape,)))
    
    # Tune the number of hidden layers and units in each.
    # Number of hidden layers: 1 - 5
    # Number of Units: 32 - 512 with stepsize of 32
    for i in range(1, hp.Int("num_layers", 2, 6)):
        model.add(
            keras.layers.Dense(
                units=hp.Int("units_" + str(i), min_value=16, max_value=256, step=16),
                activation=hp.Choice("activation", ["relu", "tanh", "linear"]),
                )
            )
        
        # Tune dropout layer with values from 0 - 0.3 with stepsize of 0.1.
        model.add(keras.layers.Dropout(hp.Float("dropout_" + str(i), 0, 0.3, step=0.1)))
    
    # Add output layer.
    model.add(keras.layers.Dense(units=1, activation="linear"))
    
    # Tune learning rate for Adam optimizer with values from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    
    # Define optimizer, loss, and metrics
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(), coeff_determination])
        
    return model


tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective="val_loss",
    max_trials=300,
    executions_per_trial=2,
    overwrite=True,
)

tuner.search(X_train, y_train, validation_data=(X_val, y_val),batch_size=32,epochs=30)


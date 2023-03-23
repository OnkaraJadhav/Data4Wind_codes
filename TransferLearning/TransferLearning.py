'''
Author: Onkar Jadhav
Python script for pre-trained model
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
#%%
# Training data
rans = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset_1B/RANS/RANS-a0_a15_a225_a30_a45_OB.txt', sep = ' ')

# Testing data for different angle
rans_ang = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/SB/RANS-a15_SB.txt', sep = ' ')

# Features
#X = rans.values[:,np.r_[1, 3, 4, 7, 8, 9]] #1, 3, 4:6, 7, 8, 9
#X_test_ang = rans_ang.values[:,np.r_[1, 3, 4, 7, 8, 9]] #1, 3, 4:6, 7, 8, 9

X = rans.values[:,1:13]
X_test_ang = rans_ang.values[:,1:13]

# Load LES data
les = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset_1B/RANS/RANS-a0_a15_a225_a30_a45_OB.txt', sep = ' ')

# Labels
y = les.values[:,4:7] # U_x

start = time.time()

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=101)

# Standardize the data
X_train, X_val, X_test_ang = standardize(X_train, X_val, X_test_ang)

## Neural Network Model
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.00001)

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
outputs = layers.Dense(3, activation='linear')(x)

# Model:
model = keras.Model(inputs,outputs, name='model')


from keras.utils.vis_utils import plot_model
plot_model(model, to_file='D:/Data4wind/VelocityStudy_v1/TransferLearning/model_plot1.png', show_shapes=True, show_layer_names=True)

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

# Fit model for train data and validate it on validation data
model.fit(X_train,y_train,
          validation_data=(X_val,y_val),
          batch_size=32,epochs=30, callbacks=[reduce_lr])
model.summary()

end = time.time()
total_time = end - start
print("\n"+ str(total_time))


# # Scores Testing
# # Load LES test data
# les_test = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/SB/LES-a15_SB.txt', sep = ' ')

# # Labels
# y_test_ang = les_test.values[:,4:5] # 
# # Model prediction
# y_pred_ang = model.predict(X_test_ang)


# # Error values (Wind tunnel):
# y_pred_ang1 = y_pred_ang[:,0:1]
# y_test_ang1 = y_test_ang[:,0:1]
# print('MAE_ang:', metrics.mean_absolute_error(y_test_ang1, y_pred_ang1))  
# print('MSE_ang:', metrics.mean_squared_error(y_test_ang1, y_pred_ang1))  
# print('RMSE_ang:', np.sqrt(metrics.mean_squared_error(y_test_ang1, y_pred_ang1)))
# print('variance:', metrics.explained_variance_score(y_test_ang1,y_pred_ang1))
# print('R2_ang:', metrics.r2_score(y_test_ang1, y_pred_ang1))

# Save model
import json

model_json = model.to_json()
with open("D:/Data4wind/VelocityStudy_v1/TransferLearning/TL_structure_MO.json", "w") as json_file:
    json.dump(model_json, json_file)
model.save_weights("D:/Data4wind/VelocityStudy_v1/TransferLearning/TL_weights_MO.h5")
###
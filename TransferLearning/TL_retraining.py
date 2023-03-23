#%%
'''
Author: Onkar Jadhav
Python script for retraining the TL model
'''
#%% Import essential libraries

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dropout

from tensorflow.keras import initializers

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from keras import backend as K

#%% R2 and standardization

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

#%%
# Training data
rans = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset_1B/RANS/RANS-a0_a45_OB.txt', sep = ' ')

# Testing data for different angle
rans_ang = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/SB/RANS-a15_SB.txt', sep = ' ')

X = rans.values[:,1:13]
X_test_ang = rans_ang.values[:,1:13]

# Load LES data
les = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset_1B/LES/LES-a0_a45_OB.txt', sep = ' ')

# Labels
y = les.values[:,4:7] # 
#y = y.ravel()

#%% Load the pre-trained model:

from tensorflow.keras.models import load_model, model_from_json
import json

with open('D:/Data4wind/VelocityStudy_v1/TransferLearning/TL_structure_MO.json','r') as f:
    model_json = json.load(f)

model = model_from_json(model_json)
model.load_weights('D:/Data4wind/VelocityStudy_v1/TransferLearning/TL_weights_MO.h5')

#%% Set how many layers you want to be trainable:
for i in range(1,6):
    model.layers[i].trainable = False

for l in model.layers:
    print(l.name, l.trainable)

# for layer in model.layers:
    # if layer.trainable:
        # print (layer.name)
#%% Add extra layers if you want:
    
initializer =  tf.keras.initializers.he_normal()

#X_rows, input_shape = np.shape(X)
#inputs = keras.Input(shape=(input_shape,)) # 
# x = layers.Dense(32, activation='tanh')(inputs)
x = layers.Dense(64,activation='tanh', kernel_initializer=initializer, name="dense_a")(model.layers[6].output) #model.layers[4].output
# x = Dropout(0.15, name="drop_a")(x)
# x = layers.Dense(64,activation='tanh', kernel_initializer=initializer,  name="dense_b")(x)
# x = Dropout(0.2, name="drop_b")(x)
# x = layers.Dense(64,activation='relu', kernel_initializer=initializer,  name="dense_c")(x)
# x = layers.Dense(128,activation='relu', kernel_initializer=initializer,  name="dense_d")(x)
outputs = layers.Dense(3, activation='linear',  name="dense_e")(x)
# outputs = layers.Dense(3, activation='linear',  name="dense_f")(model.layers[-1].output)

re_trained_model = keras.Model(inputs=model.input,outputs=outputs)
# re_trained_model = keras.Model(inputs=model.input,outputs=model.output)

re_trained_model.summary() # Print re_trained model
#%% Which layers are trainable of this new re_trained_model:

for l in re_trained_model.layers:
    print(l.name, l.trainable)
#%% Let the training begins!!!

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=2, min_lr=0.000001)

opt = keras.optimizers.Adamax(learning_rate=0.0001) # Lower learning rate makes the thing better

re_trained_model.compile(optimizer=opt,
              loss='mse',
              metrics=[tf.keras.metrics.RootMeanSquaredError(), coeff_determination]
              )

# Sort data into train, and test sets
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=101)

# Standardize the data
X_train, X_val, X_test_ang = standardize(X_train, X_val, X_test_ang)

# Fit model for train data and validate it on validation data
re_trained_model.fit(x=X_train,y=y_train,
          validation_data=(X_val,y_val),
          batch_size=32,epochs=30, callbacks=[reduce_lr])

#%% Test the new re_trained_model:
# Load LES test data
les_test = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/SB/LES-a15_SB.txt', sep = ' ')

# Labels
y_test_ang = les_test.values[:,4:7] # 
# y_test_ang = y_test_ang.ravel()

# Model prediction
y_pred_ang = re_trained_model.predict(X_test_ang)

# Error values (Wind tunnel):
y_pred_ang1 = y_pred_ang[:,0:1]
y_test_ang1 = y_test_ang[:,0:1]

print('MAE_ang:', metrics.mean_absolute_error(y_test_ang1, y_pred_ang1))  
print('MSE_ang:', metrics.mean_squared_error(y_test_ang1, y_pred_ang1))  
print('RMSE_ang:', np.sqrt(metrics.mean_squared_error(y_test_ang1, y_pred_ang1)))
print('variance:', metrics.explained_variance_score(y_test_ang1,y_pred_ang1))
print('R2_ang:', metrics.r2_score(y_test_ang1, y_pred_ang1))

# print('MAE_ang:', metrics.mean_absolute_error(y_test_ang, y_pred_ang))  
# print('MSE_ang:', metrics.mean_squared_error(y_test_ang, y_pred_ang))  
# print('RMSE_ang:', np.sqrt(metrics.mean_squared_error(y_test_ang, y_pred_ang)))
# print('VarScore_ang:', metrics.explained_variance_score(y_test_ang,y_pred_ang))
# print('R2_ang:', metrics.r2_score(y_test_ang, y_pred_ang))

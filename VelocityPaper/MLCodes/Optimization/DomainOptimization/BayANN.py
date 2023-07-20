'''
Author: Onkar Jadhav
Python script for a 4 layer artificial neural network
'''
#%%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import initializers
import time

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn import metrics

import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow_probability.python.math import random_rademacher

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

def Norml(X_train, X_test, X_test_ang):
    sc = StandardScaler()
    # Scale train features
    X_train_std = sc.fit_transform(X_train)
    # Use same scaling for test features
    X_test_std = sc.transform(X_test)
    X_test_ang_std = sc.transform(X_test_ang)
    return X_train_std, X_test_std, X_test_ang_std

# Training data
rans = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset_1B/RANS/RANS-a0_a45_OB.txt', sep = ' ')

Features = rans.columns
# Testing data for different angle
rans_ang = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/SB/RANS-a15_SB.txt', sep = ' ')

X = rans.values[:,1:13]
X_test_ang = rans_ang.values[:,1:13]

# Load LES data
les = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset_1B/LES/LES-a0_a45_OB.txt', sep = ' ')

# Labels
y = les.values[:,10:11] # U_x

start = time.time()

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=101)

# Standardize the data
X_train, X_val, X_test_ang = standardize(X_train, X_val, X_test_ang)

X_train = tf.cast(X_train, tf.float64)
X_val = tf.cast(X_val, tf.float64)
y_train = tf.cast(y_train, tf.float64)
y_val = tf.cast(y_val, tf.float64)

N_val = X_val.shape[0] 
X_rows, input_shape = np.shape(X)

data_train = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)).shuffle(10000).batch(32)

data_val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(N_val)

# Xavier initializer
def xavier(shape):
    return tf.random.truncated_normal(
        shape, 
        mean=0.0,
        stddev=np.sqrt(2/sum(shape)))

class BayesianDenseLayer(tf.keras.Model):
    """A fully-connected Bayesian neural network layer
    
    Parameters
    ----------
    d_in : int
        Dimensionality of the input (# input features)
    d_out : int
        Output dimensionality (# units in the layer)
    name : str
        Name for the layer
        
    Attributes
    ----------
    losses : tensorflow.Tensor
        Sum of the Kullback–Leibler divergences between
        the posterior distributions and their priors
        
    Methods
    -------
    call : tensorflow.Tensor
        Perform the forward pass of the data through
        the layer
    """
    
    def __init__(self, d_in, d_out, name=None):
        
        super(BayesianDenseLayer, self).__init__(name=name)
        self.d_in = d_in
        self.d_out = d_out
        
        self.w_loc = tf.Variable(xavier([d_in, d_out]), name='w_loc')
        self.w_std = tf.Variable(xavier([d_in, d_out])-6.0, name='w_std')
        self.b_loc = tf.Variable(xavier([1, d_out]), name='b_loc')
        self.b_std = tf.Variable(xavier([1, d_out])-6.0, name='b_std')
    
    
    def call(self, x, sampling=True):
        """Perform the forward pass"""
        
        if sampling:
        
            # Flipout-estimated weight samples
            s = random_rademacher(tf.shape(x))
            r = random_rademacher([x.shape[0], self.d_out])
            w_samples = tf.nn.softplus(self.w_std)*tf.random.normal([self.d_in, self.d_out])
            w_perturbations = r*tf.matmul(x*s, w_samples)
            w_outputs = tf.matmul(x, self.w_loc) + w_perturbations
            
            # Flipout-estimated bias samples
            r = random_rademacher([x.shape[0], self.d_out])
            b_samples = tf.nn.softplus(self.b_std)*tf.random.normal([self.d_out])
            b_outputs = self.b_loc + r*b_samples
            
            return w_outputs + b_outputs
        
        else:
            return x @ self.w_loc + self.b_loc
    
    
    @property
    def losses(self):
        """Sum of the KL divergences between priors + posteriors"""
        weight = tfd.Normal(self.w_loc, tf.nn.softplus(self.w_std))
        bias = tfd.Normal(self.b_loc, tf.nn.softplus(self.b_std))
        prior = tfd.Normal(0, 1)
        return (tf.reduce_sum(tfd.kl_divergence(weight, prior)) +
                tf.reduce_sum(tfd.kl_divergence(bias, prior)))
    
    
class BayesianDenseNetwork(tf.keras.Model):
    """A multilayer fully-connected Bayesian neural network
    
    Parameters
    ----------
    dims : List[int]
        List of units in each layer
    name : str
        Name for the network
        
    Attributes
    ----------
    losses : tensorflow.Tensor
        Sum of the Kullback–Leibler divergences between
        the posterior distributions and their priors, 
        over all layers in the network
        
    Methods
    -------
    call : tensorflow.Tensor
        Perform the forward pass of the data through
        the network
    """
    
    def __init__(self, dims, name=None):
        
        super(BayesianDenseNetwork, self).__init__(name=name)
        
        self.steps = []
        self.acts = []
        for i in range(len(dims)-1):
            self.steps += [BayesianDenseLayer(dims[i], dims[i+1])]
            self.acts += [tf.nn.relu]
            
        self.acts[-1] = lambda x: x
        
    
    def call(self, x, sampling=True):
        """Perform the forward pass"""

        for i in range(len(self.steps)):
            x = self.steps[i](x, sampling=sampling)
            x = self.acts[i](x)
            
        return x
    
    
    @property
    def losses(self):
        """Sum of the KL divergences between priors + posteriors"""
        return tf.reduce_sum([s.losses for s in self.steps])

#%%
class BayesianDenseRegression(tf.keras.Model):
    """A multilayer Bayesian neural network regression
    
    Parameters
    ----------
    dims : List[int]
        List of units in each layer
    name : str
        Name for the network
        
    Attributes
    ----------
    losses : tensorflow.Tensor
        Sum of the Kullback–Leibler divergences between
        the posterior distributions and their priors, 
        over all layers in the network
        
    Methods
    -------
    call : tensorflow.Tensor
        Perform the forward pass of the data through
        the network, predicting both means and stds
    log_likelihood : tensorflow.Tensor
        Compute the log likelihood of y given x
    samples : tensorflow.Tensor
        Draw multiple samples from the predictive distribution
    """    
    
    
    def __init__(self, dims, name=None):
        
        super(BayesianDenseRegression, self).__init__(name=name)
        
        # Multilayer fully-connected neural network to predict mean
        self.loc_net = BayesianDenseNetwork(dims)
        
        # Variational distribution variables for observation error
        self.std_alpha = tf.Variable([10.0], name='std_alpha')
        self.std_beta = tf.Variable([10.0], name='std_beta')

    
    def call(self, x, sampling=True):
        """Perform forward pass, predicting both means + stds"""
        
        # Predict means
        loc_preds = self.loc_net(x, sampling=sampling)
    
        # Predict std deviation
        posterior = tfd.Gamma(self.std_alpha, self.std_beta)
        transform = lambda x: tf.sqrt(tf.math.reciprocal(x))
        N = x.shape[0]
        if sampling:
            std_preds = transform(posterior.sample([N]))
        else:
            std_preds = tf.ones([N, 1])*transform(posterior.mean())
    
        # Return mean and std predictions
        return tf.concat([loc_preds, std_preds], 1)
    
    
    def log_likelihood(self, x, y, sampling=True):
        """Compute the log likelihood of y given x"""
        
        # Compute mean and std predictions
        preds = self.call(x, sampling=sampling)
        
        # Return log likelihood of true data given predictions
        return tfd.Normal(preds[:,0], preds[:,1]).log_prob(y[:,0])
    
    
    @tf.function
    def sample(self, x):
        """Draw one sample from the predictive distribution"""
        preds = self.call(x)
        return tfd.Normal(preds[:,0], preds[:,1]).sample()
    
    
    def samples(self, x, n_samples=1):
        """Draw multiple samples from the predictive distribution"""
        samples = np.zeros((x.shape[0], n_samples))
        for i in range(n_samples):
            samples[:,i] = self.sample(x)
        return samples
    
    
    @property
    def losses(self):
        """Sum of the KL divergences between priors + posteriors"""
                
        # Loss due to network weights
        net_loss = self.loc_net.losses

        # Loss due to std deviation parameter
        posterior = tfd.Gamma(self.std_alpha, self.std_beta)
        prior = tfd.Gamma(10.0, 10.0)
        std_loss = tfd.kl_divergence(posterior, prior)

        # Return the sum of both
        return net_loss + std_loss

model1 = BayesianDenseRegression([input_shape, 64, 128, 128, 64, 1])

optimizer = keras.optimizers.Adamax(learning_rate=0.001)

N = X_train.shape[0]

@tf.function
def train_step(x_data, y_data):
    with tf.GradientTape() as tape:
        log_likelihoods = model1.log_likelihood(x_data, y_data)
        kl_loss = model1.losses
        elbo_loss = kl_loss/N - tf.reduce_mean(log_likelihoods)
    gradients = tape.gradient(elbo_loss, model1.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model1.trainable_variables))
    return elbo_loss

# Fit the model
EPOCHS = 100
elbo1 = np.zeros(EPOCHS)
mae1 = np.zeros(EPOCHS)
for epoch in range(EPOCHS):
    
    # Update weights each batch
    for x_data, y_data in data_train:
        x_data = tf.cast(x_data, tf.float64)
        y_data = tf.cast(y_data, tf.float64)
        elbo1[epoch] += train_step(tf.cast(x_data, tf.float32), tf.cast(y_data, tf.float32))
        
    # Evaluate performance on validation data
    for x_data, y_data in data_val:
        y_pred = model1(x_data, sampling=False)[:, 0]
        mae1[epoch] = np.sqrt(metrics.mean_squared_error(y_pred, y_data))

# Plot the ELBO loss
plt.plot(elbo1)
plt.xlabel('Epoch')
plt.ylabel('ELBO Loss')
plt.show()

plt.plot(mae1)
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.show()


#%%
class BayesianDensityNetwork(tf.keras.Model):
    """Multilayer fully-connected Bayesian neural network, with
    two heads to predict both the mean and the standard deviation.
    
    Parameters
    ----------
    units : List[int]
        Number of output dimensions for each layer
        in the core network.
    units : List[int]
        Number of output dimensions for each layer
        in the head networks.
    name : None or str
        Name for the layer
    """
    
    
    def __init__(self, units, head_units, name=None):
        
        # Initialize
        super(BayesianDensityNetwork, self).__init__(name=name)
        
        # Create sub-networks
        self.core_net = BayesianDenseNetwork(units)
        self.loc_net = BayesianDenseNetwork([units[-1]]+head_units)
        self.std_net = BayesianDenseNetwork([units[-1]]+head_units)

    
    def call(self, x, sampling=True):
        """Pass data through the model
        
        Parameters
        ----------
        x : tf.Tensor
            Input data
        sampling : bool
            Whether to sample parameter values from their 
            variational distributions (if True, the default), or
            just use the Maximum a Posteriori parameter value
            estimates (if False).
            
        Returns
        -------
        preds : tf.Tensor of shape (Nsamples, 2)
            Output of this model, the predictions.  First column is
            the mean predictions, and second column is the standard
            deviation predictions.
        """
        
        # Pass data through core network
        x = self.core_net(x, sampling=sampling)
        x = tf.nn.relu(x)
        
        # Make predictions with each head network
        loc_preds = self.loc_net(x, sampling=sampling)
        std_preds = self.std_net(x, sampling=sampling)
        std_preds = tf.nn.softplus(std_preds)
        
        # Return mean and std predictions
        return tf.concat([loc_preds, std_preds], 1)
    
    
    def log_likelihood(self, x, y, sampling=True):
        """Compute the log likelihood of y given x"""
        
        # Compute mean and std predictions
        preds = self.call(x, sampling=sampling)
        
        # Return log likelihood of true data given predictions
        return tfd.Normal(preds[:,0], preds[:,1]).log_prob(y[:,0])
        
        
    @tf.function
    def sample(self, x):
        """Draw one sample from the predictive distribution"""
        preds = self.call(x)
        return tfd.Normal(preds[:,0], preds[:,1]).sample()
    
    
    def samples(self, x, n_samples=1):
        """Draw multiple samples from predictive distributions"""
        samples = np.zeros((x.shape[0], n_samples))
        for i in range(n_samples):
            samples[:,i] = self.sample(x)
        return samples
    
    
    @property
    def losses(self):
        """Sum of the KL divergences between priors + posteriors"""
        return (self.core_net.losses +
                self.loc_net.losses +
                self.std_net.losses)


model2 = BayesianDensityNetwork([input_shape, 32, 64, 64], [64, 64, 32, 1])
optimizer = keras.optimizers.Adamax(learning_rate=0.0001)

N = X_train.shape[0]

@tf.function
def train_step(x_data, y_data):
    with tf.GradientTape() as tape:
        log_likelihoods = model2.log_likelihood(x_data, y_data)
        kl_loss = model2.losses
        elbo_loss = kl_loss/N - tf.reduce_mean(log_likelihoods)
    gradients = tape.gradient(elbo_loss, model2.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model2.trainable_variables))
    return elbo_loss

# Fit the model
EPOCHS = 100
elbo2 = np.zeros(EPOCHS)
mae2 = np.zeros(EPOCHS)
for epoch in range(EPOCHS):
    
    # Update weights each batch
    for x_data, y_data in data_train:
        x_data = tf.cast(x_data, tf.float64)
        y_data = tf.cast(y_data, tf.float64)
        elbo2[epoch] += train_step(tf.cast(x_data, tf.float32), tf.cast(y_data, tf.float32))
        
    # Evaluate performance on validation data
    for x_data, y_data in data_val:
        y_pred = model2(x_data, sampling=False)[:, 0]
        mae2[epoch] = np.sqrt(metrics.mean_squared_error(y_pred, y_data))

# Plot the ELBO loss
plt.plot(elbo2)
plt.xlabel('Epoch')
plt.ylabel('ELBO Loss')
plt.show()

plt.plot(mae2)
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.show()


#%%
# Scores Testing
# Load LES test data
les_test = pd.read_csv('D:/Data4wind/VelocityStudy_v1/Dataset2/SB/LES-a15_SB.txt', sep = ' ')
# Labels
y_test_ang = les_test.values[:,10:11] # CpPrime

# Model prediction
y_pred_ang = model2.predict(X_test_ang)


# Error values (Wind tunnel):
y_pred_ang1 = y_pred_ang[:,0:1]
y_test_ang1 = y_test_ang[:,0:1]
print('MAE_ang:', metrics.mean_absolute_error(y_test_ang1, y_pred_ang1))  
print('MSE_ang:', metrics.mean_squared_error(y_test_ang1, y_pred_ang1))  
print('RMSE_ang:', np.sqrt(metrics.mean_squared_error(y_test_ang1, y_pred_ang1)))
print('variance:', metrics.explained_variance_score(y_test_ang1,y_pred_ang1))
print('R2_ang:', metrics.r2_score(y_test_ang1, y_pred_ang1))
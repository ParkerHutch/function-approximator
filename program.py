# -*- coding: utf-8 -*-
"""
@author: Parker Hutchinson

LESSONS LEARNED:
- TensorFlow neural networks don't require tf.keras.data.Dataset objects
    for training
- Scaling the data improved the training performance and duration greatly
    for this dataset
- Using an EarlyStopper sometimes prevented the model from fitting the data
    well - when the model was allowed to train for the full >100 epochs with
    enough neurons, it could usually make a nice-looking curve
- Use the right metrics. The accuracy metric was originally used while training
    the model, but this value is a poor indicator for performance in regression
    models.
"""
import numpy as np
import pandas as pd

import tensorflow as tf

import random

# Set random seeds for consistent results
np.random.seed(1)
tf.random.set_seed(1)
random.seed(1)

""" DATA CREATION """    
def target_function(x):
    """
    Produce a y-coordinate from the input x-coordinate, determined by a 
    predefined mathematical function. 

    Parameters
    ----------
    x : float
        An input x-coordinate

    Returns
    -------
    float
        The output of the predefined mathematical function at the input
        x-coordinate.

    """
    return x**3.0

df = pd.DataFrame()
df['X'] = np.linspace(-10,10,500)
df['Y'] = [target_function(x) for x in df['X'].values]

""" DATA PREPROCESSING """
from sklearn.preprocessing import MinMaxScaler

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

df['X'] = x_scaler.fit_transform(df['X'].values.reshape(-1, 1))
df['Y'] = y_scaler.fit_transform(df['Y'].values.reshape(-1, 1))

""" DATA SPLITTING """
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['X'], df['Y'], 
                                                    test_size=.2,
                                                    random_state=1)
""" MODEL CREATION """
def build_model(num_hidden_layers, num_hidden_units):
    """
    Construct a neural network with the specified number of hidden layers and 
    hidden units. 

    Parameters
    ----------
    num_hidden_layers : int
        The number of neuron layers between the model's input and output 
        layers.
    num_hidden_units : int
        The number of neurons to be placed in each of the model's hidden
        layers.

    Returns
    -------
    model : tf.keras.Sequential
        An uncompiled neural network of size 
        num_hidden_layers * num_hidden_units + 2 and depth
        num_hidden_layers + 2.

    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(num_hidden_units, input_dim=1, 
                                    activation='relu'))
    for i in range(num_hidden_layers - 1):
        model.add(tf.keras.layers.Dense(num_hidden_units, activation='relu'))
    
    model.add(tf.keras.layers.Dense(1))
    return model

model = build_model(3, 50)

model.compile(loss='mse', optimizer='adam')
# Try different optimizers: with SGD, loss goes to infinity
# MSE is good for regression

""" TRAINING """
max_epochs = 100
batch_size = 3
validation_split = 0.1 
model.fit(X_train, y_train, epochs=max_epochs, batch_size=batch_size,
          validation_split = validation_split,
          verbose=1)

""" EVALUATION """
test_loss = model.evaluate(X_test, y_test)

eval_df = pd.DataFrame({'X': X_test, 
                        'Target': y_test, 
                        'Prediction': 
                        [pred[0] for pred in model.predict(X_test)]})

# Unscale the values in the DataFrame
eval_df['X'] = x_scaler.inverse_transform(
    eval_df['X'].values.reshape(-1, 1))
eval_df['Target'] = y_scaler.inverse_transform(
    eval_df['Target'].values.reshape(-1, 1))
eval_df['Prediction'] = y_scaler.inverse_transform(
    eval_df['Prediction'].values.reshape(-1, 1))

eval_df.sort_values(by=['X'], inplace=True)

import matplotlib.pyplot as plt
plt.plot(eval_df['X'], eval_df['Target'])
plt.plot(eval_df['X'], eval_df['Prediction'])
plt.show()

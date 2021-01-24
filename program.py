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
- Calling model.fit multiple times will fit the model from its current weights,
    without resetting them (i.e. as if you were fitting a new model)
- Different optimizers and loss functions can make a difference in training
    duration and performance (Adam optimizer and mean-squared-error loss 
    seem best for this case)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

x_range = (-10, 10)
num_points = 500
df = pd.DataFrame()
df['X'] = np.linspace(x_range[0], x_range[1], num_points)
df['Y'] = [target_function(x) for x in df['X'].values]

""" DATA PREPROCESSING """
from sklearn.preprocessing import MinMaxScaler

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

df['X'] = x_scaler.fit_transform(df['X'].values.reshape(-1, 1))
df['Y'] = y_scaler.fit_transform(df['Y'].values.reshape(-1, 1))

""" DATA SPLITTING """
test_split = 0.2
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['X'], df['Y'], 
                                                    test_size=test_split,
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
print('Training model')
max_epochs = 100
batch_size = 3
validation_split = 0.1 
# The history object stores training information for loss and any metrics in 
# the model configuration 
history = model.fit(X_train, y_train, epochs=max_epochs, batch_size=batch_size,
          validation_split = validation_split,
          verbose=0)
loss_history = history.history['loss']

plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Set Loss versus Epoch')
plt.show()


""" EVALUATION """
test_loss = model.evaluate(X_test, y_test, verbose=0)

eval_df = pd.DataFrame({'X': X_test, 'Target': y_test, 
     'Prediction': model.predict(X_test).flatten().astype(np.float64)
     })

# Unscale the values in the DataFrame
eval_df['X'] = x_scaler.inverse_transform(
    eval_df['X'].values.reshape(-1, 1))
eval_df['Target'] = y_scaler.inverse_transform(
    eval_df['Target'].values.reshape(-1, 1))
eval_df['Prediction'] = y_scaler.inverse_transform(
    eval_df['Prediction'].values.reshape(-1, 1))

eval_df.sort_values(by=['X'], inplace=True)

plt.plot(eval_df['X'], eval_df['Target'], label='Target')
plt.plot(eval_df['X'], eval_df['Prediction'], label='Prediction')
plt.legend(loc='upper left')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Model Predictions versus Target Values')
plt.show()

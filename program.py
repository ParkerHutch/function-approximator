# -*- coding: utf-8 -*-
"""
@author: Parker Hutchinson
"""
import numpy as np
import pandas as pd

import random
import math

import tensorflow as tf

np.random.seed(1)
tf.random.set_seed(1)
random.seed(1)

""" DATA CREATION """    
def target_function(x):
    return x ** 2.0

df = pd.DataFrame()
df['X'] = [i for i in range(-5000, 5000)]
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
# Split train set to create validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                  test_size=.1,
                                                  random_state=1)
    
""" MODEL CREATION """
def build_model(num_hidden_layers, num_hidden_units):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(num_hidden_units, input_dim=1, 
                                    activation='relu'))
    for i in range(num_hidden_layers - 1):
        model.add(tf.keras.layers.Dense(num_hidden_units, activation='relu'))
    
    model.add(tf.keras.layers.Dense(1))
    return model

model = build_model(2, 10)

#optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
model.compile(loss='mse', optimizer='adam')
# Try different optimizers: with SGD, loss goes to infinity
# MSE is good for regression

""" TRAINING """
num_epochs=20
early_stopper = tf.keras.callbacks.EarlyStopping(monitor='mae', 
                                                 mode='min', patience=2)
model.fit(X_train, y_train, batch_size=10, epochs=num_epochs)

""" EVALUATION """
test_loss = model.evaluate(X_test, y_test)

eval_df = pd.DataFrame({'X':X_test, 
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

import seaborn as sns
import matplotlib.pyplot as plt
ax = sns.lineplot(data=eval_df[['Target', 'Prediction']])

# -*- coding: utf-8 -*-
"""
@author: Parker Hutchinson
"""
import numpy as np
import pandas as pd

import random
import math

import tensorflow as tf

""" DATA CREATION """
def noise(variance):
    return random.randrange(-variance, variance)
    
def target_function(x):
    return 2 * x
    #return 5 * x + 2 + noise(2)

df = pd.DataFrame()
df['X'] = [x for x in range(0, 1000)]
df['Y'] = df['X'].apply(target_function)

""" DATA PREPROCESSING """ 

"""
TODO: Do I really need normalization for this?

# Normalize the data using a MinMaxScaler

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df['X'] = scaler.fit_transform(df[['X']])
df['Y'] = scaler.fit_transform(df[['Y']])
"""

""" MODEL CREATION """
def build_model(num_layers, num_hidden_units):
    model = tf.keras.Sequential()
    for i in range(num_layers):
        model.add(tf.keras.layers.Dense(num_hidden_units, activation='relu'))
    
    model.add(tf.keras.layers.Dense(1))
    return model

# Simple model: build_model(1, 2)
model = build_model(2, 50)

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# Try different optimizers: with SGD, loss goes to infinity

num_epochs=20
early_stopper = tf.keras.callbacks.EarlyStopping(monitor='mae', 
                                                 mode='min', patience=2)

model.fit(df['X'].values, df['Y'].values, epochs=num_epochs, 
          callbacks=[early_stopper])

""" EVALUATION """
preds = [pred[0] for pred in model.predict(df['X']).tolist()]

import seaborn as sns
import matplotlib.pyplot as plt
ax = sns.lineplot(x=df['X'][:100], y=preds[:100])
ax = sns.lineplot(x=df['X'][:100], y=df['Y'][:100])


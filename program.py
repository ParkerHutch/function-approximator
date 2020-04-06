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
def noise(variance):
    return random.randrange(-variance, variance)
    
def target_function(x):
    return 2 * x

df = pd.DataFrame()
df['X'] = [x for x in range(0, 1000)]
df['Y'] = df['X'].apply(target_function)

""" MODEL CREATION """
def build_model(num_layers, num_hidden_units):
    model = tf.keras.Sequential()
    for i in range(num_layers):
        model.add(tf.keras.layers.Dense(num_hidden_units, activation='relu'))
    
    model.add(tf.keras.layers.Dense(1))
    return model

# Simple model: build_model(1, 2)
model = build_model(2, 50)

optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
model.compile(loss='huber_loss', optimizer=optimizer, metrics=['mae'])
# Try different optimizers: with SGD, loss goes to infinity

num_epochs=10
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


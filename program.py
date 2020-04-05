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
    return 5 * x + 2

df = pd.DataFrame()
df['X'] = [x for x in range(0, 100000)]
df['Y'] = df['X'].apply(target_function)

""" DATA PREPROCESSING """ 

"""
TODO: Do I really need normalization for this?

# Normalize the data using a MinMaxScaler

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df['X'] = scaler.fit_transform(df[['X']])
df['Y'] = scaler.fit_transform(df[['Y']])"""

""" MODEL CREATION """
def build_model(num_layers, num_hidden_units):
    model = tf.keras.Sequential()
    for i in range(num_layers):
        model.add(tf.keras.layers.Dense(num_hidden_units, activation='relu'))
    
    model.add(tf.keras.layers.Dense(1))
    return model

# Simple model: build_model(1, 2)
model = build_model(2, 50)
"""
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(1)
    ])
"""

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# Try different optimizers: SGD goes to infinity

num_epochs=20
early_stopper = tf.keras.callbacks.EarlyStopping(monitor='loss', 
                                                 mode='min', patience=3)
model.fit(df['X'].values, df['Y'].values, epochs=num_epochs, 
          callbacks=[early_stopper])



""" EVALUATION """
cleaned_y = []
for val in preds:
    for cleaned_val in val:
        cleaned_y.append(cleaned_val)
        
"""
for element in train_dataset.as_numpy_iterator(): 
    for val in element:
        print(val)"""
        
import seaborn as sns
import matplotlib.pyplot as plt
ax = sns.lineplot(x=data['X'], y=cleaned_y)
ax = sns.lineplot(x=data['X'], y=data['Y'])




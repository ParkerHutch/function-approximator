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
    #return 2 * x + noise(2)

# TODO SIMPLIFY
df = pd.DataFrame()
df['X'] = [x for x in range(0, 100000)]
df['Y'] = df['X'].apply(target_function)

"""
data = {'X':[], 'Y':[]}
for i in range(0, 10000):
    data['X'].append(i)
    data['Y'].append(target_function(i))

df = pd.DataFrame(data)
"""

""" DATA PREPROCESSING """ 

"""

TODO: Do I really need normalization for this?

"""
# Normalize the data using a MinMaxScaler
"""
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df['X'] = scaler.fit_transform(df[['X']])
df['Y'] = scaler.fit_transform(df[['Y']])"""


""" DATA CONVERSION """

# Convert the data to the Tensorflow format
#target = df.pop('Y')
#dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

#train_dataset = dataset.shuffle(len(df)).batch(10)

""" MODEL CREATION """
# Hyperparemeters
hidden_layer_size = 2
output_size = 1
num_epochs=20

early_stopper = tf.keras.callbacks.EarlyStopping(monitor='loss', 
                                                 mode='min', patience=2)

""" SIMPLE MODEL: 2 hidden units
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dense(output_size)
    ])
"""

model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(output_size)
    ])

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# Try different optimizers: SGD goes to infinity

model.fit(data['X'], data['Y'], epochs=num_epochs, callbacks=[early_stopper])

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




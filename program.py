# -*- coding: utf-8 -*-
"""
@author: Parker Hutchinson
"""
import numpy as np
import pandas as pd

import random
import math
""" DATA CREATION """
def noise(variance):
    return random.randrange(-variance, variance)
    
def target_function(x):
    return math.sin(x)
    #return 2 * x + noise(2)

data = {'X':[], 'Y':[]}
for i in range(0, 1000):
    data['X'].append(i)
    data['Y'].append(target_function(i))

df = pd.DataFrame(data)

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
import tensorflow as tf

# Convert the data to the Tensorflow format
target = df.pop('Y')
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

#train_dataset = dataset.shuffle(len(df))
#train_dataset = dataset.shuffle(len(df)).batch(5)



""" MODEL CREATION """
# Hyperparemeters
hidden_layer_size = 10
output_size = 1
num_epochs=20
"""
early_stopper = tf.keras.callbacks.EarlyStopping(monitor='accuracy', 
                                                 mode='max', patience=2)
"""

model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, input_dim=1, activation='linear'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='linear')
    ])

model.compile(loss='mae', metrics=['mae', 'mse'])

"""
def build_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
model = build_model()"""

model.fit(train_dataset, epochs=num_epochs)

preds = model.predict(train_dataset)



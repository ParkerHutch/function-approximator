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
    return 5 * x + 2
    #return 2 * x + noise(2)

data = {'X':[], 'Y':[]}
for i in range(0, 10000):
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

train_dataset = dataset.shuffle(len(df))
train_dataset = dataset.shuffle(len(df)).batch(5)



""" MODEL CREATION """
# Hyperparemeters
hidden_layer_size = 10
output_size = 1
num_epochs=10
"""
early_stopper = tf.keras.callbacks.EarlyStopping(monitor='accuracy', 
                                                 mode='max', patience=2)
"""

model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, input_shape=(1,), activation='linear'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(output_size)
    ])

model.compile(loss='mse', metrics=['mae'])

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

features, labels = next(iter(train_dataset))
for feat, arg in train_dataset:
    #print(feat)
    
model.fit(train_dataset, epochs=num_epochs)

new_data = {'X':data['X'], 'Y':model.predict(data['X']).tolist()}

new_df = pd.DataFrame(new_data)

preds = model.predict(data['X'])
#preds = model.predict(train_dataset)

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


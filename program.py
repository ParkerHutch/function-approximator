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
def build_model(num_layers, num_hidden_units):
    model = tf.keras.Sequential()
    for i in range(num_layers):
        model.add(tf.keras.layers.Dense(num_hidden_units, activation='relu'))
    
    model.add(tf.keras.layers.Dense(1))
    return model

# Simple model: build_model(1, 2)
model = build_model(1, 50)

optimizer = tf.keras.optimizers.Adam(learning_rate=.01)
model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
# Try different optimizers: with SGD, loss goes to infinity

num_epochs=20
early_stopper = tf.keras.callbacks.EarlyStopping(monitor='mae', 
                                                 mode='min', patience=20)
model.fit(X_train.values, y_train.values, epochs=num_epochs, 
          callbacks=[early_stopper])

""" EVALUATION """
test_loss, test_acc = model.evaluate(X_test, y_test)

test_df = pd.DataFrame({'X':X_test, 'Target': y_test, 'Prediction': 
                        [pred[0] for pred in model.predict(X_test)]})

import seaborn as sns
import matplotlib.pyplot as plt
ax = sns.lineplot(data=test_df[['Target', 'Prediction']])


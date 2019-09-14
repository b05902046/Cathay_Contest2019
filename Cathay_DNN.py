#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('newtrain.csv', 
                 encoding = 'utf-8')

target = pd.DataFrame(df['Y1'])
features = df.drop(['Y1', 'CUS_ID'], axis = 1)

x_train,x_test, y_train, y_test = train_test_split(features, 
                                                   target, 
                                                   test_size = 0.3, 
                                                   random_state = 42)
x_train = pd.get_dummies(x_train)
x_train = x_train.values
y_train = y_train['Y1'].map({'N':0, 'Y':1})
y_train = y_train.values

x_test = pd.get_dummies(x_test)
x_test = x_test.values
y_test = y_test['Y1'].map({'N':0, 'Y':1})
y_test = y_test.values

from sklearn import decomposition
pca = decomposition.PCA(n_components=100)
#x_train = x_train.dropna()
pca.fit(x_train)
pca.fit(x_test)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

print(x_train.shape)

import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(100, activation = tf.nn.relu),
    tf.keras.layers.Dropout(0.2), #固定把使用過的神經元丟掉(避免overfitting)
    tf.keras.layers.Dense(500, activation = tf.nn.relu), #第二層神經元 
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation = tf.nn.softmax) #分類型-->softmax，神經元數量要跟預測種類的數量一樣
])
adam = tf.keras.optimizers.Adam(lr = 0.001) #優化器(learning rate)
model.compile(optimizer = adam, #'sgd'
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])
history = model.fit(x_train, y_train, epochs = 20, validation_split = 0.3) #validation_split可不寫

model.evaluate(x_test, y_test)

y_pred = model.predict_classes(x_test)
print(y_pred)

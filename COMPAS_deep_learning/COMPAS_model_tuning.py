from __future__ import print_function
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras import regularizers
from keras import backend as K
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random
import requests
import zipfile
import io

# Fix random seed for reproducibility.
np.random.seed(1337)

# Download and extract data.
r = requests.get("http://web.stanford.edu/class/cs21si/resources/unit3_resources.zip")
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

data = pd.read_csv("unit3_resources/compas-scores.csv", header = 0)

# Select fields we want.
fields_of_interest = ['name', 'sex', 'age', 'race', 'priors_count', 'c_charge_desc', 
                      'v_decile_score', 'decile_score', 'is_violent_recid', 'is_recid']
data = data[fields_of_interest]
# More interpretable column names.
data.columns = ['name', 'sex', 'age', 'race', 'num_priors', 'charge', 
                'violence_score', 'recidivism_score', 'violence_true', 'recidivism_true']

# Remove records with missing scores.
data = data.loc[(data.violence_score != -1) & (data.recidivism_score != -1)]
data = data.loc[(data.violence_true != -1) & (data.recidivism_true != -1)]

# Convert strings to numerical values.
sex_classes = {'Male': 0, 'Female' : 1}

processed_data = data.copy()
processed_data['sex'] = data['sex'].apply(lambda x: sex_classes[x])

# One-hot encode race.
processed_data = pd.get_dummies(processed_data, columns = ['race'])
columns = processed_data.columns.tolist()
columns = columns[0:3] + columns[9:] + columns[3:9]
processed_data = processed_data.reindex(columns = columns)

# Convert pandas dataframe to numpy array for easier processing.
processed_data = processed_data.values

# split into input (X) and output (Y) variables
X = processed_data[:,1:10].astype('float32') # sex, age, race, num_priors
y = processed_data[:,14].astype('float32') # recidivism_true

num_train = int(math.ceil(X.shape[0]*0.8))
num_test = int(math.floor(X.shape[0]*0.2))

X_train = X[:num_train]
y_train = y[:num_train]

X_test = X[num_train:]
y_test = y[num_train:]

num_classes = 2
# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

#########################################################
# Trains and evaluates given model. Returns loss and 
# accuracy.
#########################################################
def eval(model, verb = 2):
    # fit the model
    model.fit(X_train, y_train, 
              epochs = 30, 
              batch_size = batch_size,          
              validation_split = 0.1,
              verbose = verb,
              shuffle = False)
    
    # Evaluate the model.
    scores = model.evaluate(X_test, y_test)
    
    return scores

batch_size = 64
num_classes = 2

learning_rate = 2e-3
reg_strength = 1e-2

#dropout_strength = 1e-2

#########################################################
# Initializes neural network with dropout.
#########################################################
def nn_classifier(learning_rate, reg_strength, dropout_strength=0.5):
    # create model
    model = Sequential()

    # Add a layer to model which has:
    # Input size: 9; and output size: 1
    model_dropout.add(Dropout(reg_strength, input_shape = (X.shape[1],)))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(100, activation = 'relu')) 
    model.add(Dense(50, activation = 'relu'))  
    model.add(Dense(num_classes, activation='softmax'))
    
    # compile model
    sgd = tf.keras.optimizers.SGD(lr = learning_rate)
    model.compile(loss = keras.losses.categorical_crossentropy, 
                  optimizer = sgd, metrics=['accuracy'])
    
    return model

# Evaluate your model
for learning_rate in [1e-2]:
  for reg_strength in [1e-4]:
    print("Using learning rate %f and regularization strength %f..." % (learning_rate, reg_strength))
    model = nn_classifier(learning_rate, reg_strength)
    loss, acc = eval(model, verb = 2)
    print('\n\nTest loss:', loss)
    print('Test accuracy:', acc)
    
dropout_strength = 1e-2
batch_size = 64
num_classes = 2

learning_rate = 2e-3
reg_strength = 1e-4

#########################################################
# Initializes neural network with dropout.
#########################################################
def dropout_classifier(learning_rate, dropout_strength):
    # create model
    model_dropout = Sequential()

    model_dropout.add(Dropout(dropout_strength, input_shape = (X.shape[1],)))
    model_dropout.add(Dense(50, activation = 'relu')) 
    model_dropout.add(Dense(100, activation = 'relu')) 
    model_dropout.add(Dense(50, activation = 'relu')) 
    model_dropout.add(Dense(num_classes, activation = 'softmax'))
    
    # compile model
    sgd = keras.optimizers.SGD(lr = learning_rate)
    model_dropout.compile(loss = keras.losses.categorical_crossentropy, 
                  optimizer = sgd, metrics=['accuracy'])
    
    return model_dropout

model_dropout = dropout_classifier(learning_rate, dropout_strength)

loss, acc = eval(model_dropout, verb = 0)
print('\n\nTest loss:', loss)
print('Test accuracy:', acc)

def tune_hyperparams():
    best_model = (None, None, None)
    running_best_accuracy = 0
    
    learning_rate = [1, 1e-1, 1e-2, 1e-8, 2e-3]
    reg_strength = [1e-17, 1e-20, 1e-21, 1e-4, 1e-2] 
    
    for lr in learning_rate:
        for reg in reg_strength:
            model = nn_classifier(lr, reg)
            model_loss, model_acc = eval(model, verb = 0)

            print('\n val_acc: {:f}, lr: {:f}, reg: {:f}\n'.format(
                    model_acc, lr, reg))

            if model_acc > running_best_accuracy:
                model_params = {"lr": lr, "reg": reg}
                best_model = (model, model_acc, model_params)
                running_best_accuracy = model_acc
            
    return best_model
        
best_model = tune_hyperparams()
print("\n\nBest Model Performance: ", best_model[1])
print("Hyperparameters of Best Model: ", best_model[2])


  


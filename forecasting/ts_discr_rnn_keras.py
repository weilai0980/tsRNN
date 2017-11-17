#!/usr/bin/python

# data processing packages
import numpy as np   
import pandas as pd 

from scipy import stats # look at scipy
from scipy import linalg
from scipy import *

import random

# machine leanring packages
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from keras.models import Sequential 
from keras.layers.core import Dense, Activation, Dropout, Reshape, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.optimizers import *
from keras.callbacks import *

from keras.regularizers import *
from keras.initializers import *
from keras.activations import *

from utils_keras import *
from utils_dataPrepro import *

# --- training log --- 
res_file = "res/tsRnn_discr_stateful.txt"
filepath="res/model/tsRnn_discr_stateful-{epoch:02d}.hdf5"

# --- network set-up ---
para_hidden_neurons = 256
para_batch_size = 1
para_is_stateful = True
para_epochs = 500
para_lr = 0.0005
para_shuffle = False

#--- data files ---
# # air-quality data
files_list=["../../dataset/dataset_ts/air_xtrain.dat", \
            "../../dataset/dataset_ts/air_xtest.dat",\
            "../../dataset/dataset_ts/air_ytrain.dat", \
            "../../dataset/dataset_ts/air_ytest.dat"]


# Stateless - validation on each epoch 
class TestCallback_stateless(Callback):
    def __init__(self, test_data, train_data):
        self.test_data  = test_data
        self.train_data = train_data

    def on_epoch_end(self, epoch, logs={}):
        x_ts, y_ts = self.test_data
        x_tr, y_tr = self.train_data
        
        py_tr = self.model.predict(x_tr, verbose=0)
        py_ts = self.model.predict(x_ts, verbose=0)
        
       #training and testing errors 
        size_tr = len(x_tr)
        size_ts = len(x_ts)
        err_tr = [ (y_tr[i][0] - py_tr[i][0])**2 for i in range(size_tr) ]
        err_ts = [ (y_ts[i][0] - py_ts[i][0])**2 for i in range(size_ts) ]
        
        #loss
        loss = self.model.evaluate(x_tr, y_tr, verbose=0)
        
        with open(res_file, "a") as text_file:
            text_file.write("At epoch %d: loss %f, train %f, test %f\n" % ( epoch, loss, sqrt(mean(err_tr)),\
                                                                           sqrt(mean(err_ts))) )        
#        print('\nTesting loss: {}, test_err:{}, train_err:{} \n'.format(\
#               loss, sqrt( mean(err_ts) ), sqrt( mean(err_tr) ) ))

# Stateful - validation on each epoch 
class TestCallback_stateful(Callback):
    def __init__(self, test_data, train_data):
        self.test_data  = test_data
        self.train_data = train_data

    def on_epoch_end(self, epoch, logs={}):
        x_ts, y_ts = self.test_data
        x_tr, y_tr = self.train_data
        
        tmp_py=[]
        for tmp_tr in x_tr:
            tmp_tr = np.expand_dims(tmp_tr, axis=0)
            tmp_py.append( self.model.predict(tmp_tr, verbose=0) )
            
        err_tr = [ (y_tr[ i[0] ][0] - i[1])**2 for i in enumerate(tmp_py) ]
        
        tmp_py=[]
        for tmp_ts in x_ts:
            tmp_ts = np.expand_dims(tmp_ts, axis=0)
            tmp_py.append( self.model.predict(tmp_ts, verbose=0) )
            
        err_ts = [ (y_ts[ i[0] ][0] - i[1])**2 for i in enumerate(tmp_py) ]
        
        #with open(res_file, "a") as text_file:
        #    text_file.write("At epoch %d: loss %f, train %f, test %f\n" % ( epoch, loss, sqrt(mean(err_tr)),\
                                                                           #sqrt(mean(err_ts))) )   
        print('\n train_err:{}, test_err:{} \n'.format( sqrt(mean(err_tr)), sqrt(mean(err_ts)) ))



# --- load data and prepare data --- 
xtrain, ytrain, xtest, ytest, tr_shape, ts_shape = \
prepare_train_test_data( False, files_list)

print np.shape(xtrain), np.shape(ytrain), np.shape(xtest), np.shape(ytest)

# automatically format the dimensions for univairate or multi-variate cases 
if len(tr_shape)==2:
    xtrain = np.expand_dims( xtrain, 2 )
    xtest  = np.expand_dims( xtest,  2 )
    
elif len(tr_shape)==3:
    xtrain = np.reshape( xtrain, tr_shape )
    xtest  = np.reshape( xtest,  ts_shape )

ytrain = np.expand_dims( ytrain, 1 ) 
ytest  = np.expand_dims( ytest,  1 )

print np.shape(xtrain), np.shape(ytrain), np.shape(xtest), np.shape(ytest)

in_out_neurons = np.shape(xtrain)[-1]
win_size = np.shape(xtrain)[1]

# clean logs
with open(res_file, "w") as text_file:
    text_file.close()

    
# --- plain discriminative --- 

#optimizer
sgd  = SGD(lr = para_lr, momentum = 0.9, nesterov = True)
rms  = RMSprop(lr = 0.05,  rho = 0.9, epsilon  = 1e-08, decay = 0.0)
adam = Adam(lr = para_lr, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0)


model = Sequential()
model.add( LSTM(para_hidden_neurons, \
                input_dim = in_out_neurons, \
                return_sequences = False, \
                batch_input_shape = [para_batch_size, win_size, in_out_neurons ], \
#                input_shape = [para_batch_size, win_size, in_out_neurons ], \
#               input_length = win_size, \
                activation ='tanh',\
#               dropout = 0.1,\
#               kernel_regularizer = l2(0.2), 
#               recurrent_regularizer = l2(0.1),\
#               !change!
                stateful = para_is_stateful, \
                kernel_initializer    = glorot_normal(), \
                recurrent_initializer = glorot_normal() ))
# change: activiation
'''
model.add( LSTM(hidden_neurons, return_sequences = False, \
 #               input_shape = [batch_size, win_size, in_out_neurons ], \
 #               input_length = win_size, \
               activation ='tanh',\
 #               dropout = 0.1,\
 #               kernel_regularizer = l2(0.2), 
 #               recurrent_regularizer = l2(0.1),\
 #               !change!
               kernel_initializer    = glorot_normal(), \
               recurrent_initializer = glorot_normal() ))
'''

# identity ini
# he's ini:   he_normal()
# xavier ini: glorot_normal()
# orthogonal: Orthogonal()

model.add(Dropout(0.2))
model.add(Dense(128,\
                    kernel_regularizer = l2(0.001), \
                    kernel_initializer = he_normal()))
model.add(Activation("relu"))
model.add(Dense(64, \
#                    kernel_regularizer = l2(0.001), \
                    kernel_initializer = he_normal()))
model.add(Activation("relu"))
model.add(Dense(32, \
#                    kernel_regularizer = l2(0.001), \
                    kernel_initializer = he_normal()))
model.add(Activation("relu"))

model.add(Dense(1,  \
#                    kernel_regularizer = l2(0.01), \
                    kernel_initializer = he_normal()))
model.add(Activation("linear"))

# save model 
checkpointer = ModelCheckpoint(filepath, verbose=0, save_best_only=False, save_weights_only=False,\
                               period = 10 )

# training
model.compile(loss = "mean_squared_error", optimizer = adam)

if para_is_stateful:
    
    for i in range(para_epochs):
        model.fit( xtrain, ytrain, shuffle = para_shuffle, \
           callbacks = [ \
           TestCallback_stateful( (xtest, ytest), (xtrain, ytrain) ), checkpointer ], \
           batch_size = para_batch_size, epochs = 1)
        
        model.reset_states()
    
else:
    model.fit( xtrain, ytrain, shuffle = para_shuffle, \
           callbacks = [ \
           TestCallback_stateless( (xtest, ytest), (xtrain, ytrain) ), checkpointer ], \
           batch_size = para_batch_size, epochs = 500)

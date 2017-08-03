
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



# validation on each epoch 
class TestCallback(Callback):
    def __init__(self, test_data, train_data):
        self.test_data  = test_data
        self.train_data = train_data

    def on_epoch_end(self, epoch, logs={}):
        x_ts, y_ts = self.test_data
        x_tr, y_tr = self.train_data
        
        py_tr = self.model.predict(x_tr, verbose=0)
        py_ts = self.model.predict(x_ts, verbose=0)
        
        size_tr = len(x_tr)
        size_ts = len(x_ts)
        
        err_tr = [ (y_tr[i][0] - py_tr[i][0])**2 for i in range(size_tr) ]
        err_ts = [ (y_ts[i][0] - py_ts[i][0])**2 for i in range(size_ts) ]
        
        loss = self.model.evaluate(x_ts, y_ts, verbose=0)
        
        with open("res/tsRnn.txt", "a") as text_file:
            text_file.write("At epoch %d: loss %f, train %f, test %f\n" % ( epoch, loss, sqrt(mean(err_tr)),\
                                                                           sqrt(mean(err_ts))) )        
#        print('\nTesting loss: {}, test_err:{}, train_err:{} \n'.format(\
#               loss, sqrt( mean(err_ts) ), sqrt( mean(err_tr) ) ))


# # air-quality data

files_list=["../../dataset/dataset_ts/air_xtrain.dat", \
            "../../dataset/dataset_ts/air_xtest.dat",\
            "../../dataset/dataset_ts/air_ytrain.dat", \
            "../../dataset/dataset_ts/air_ytest.dat"]

xtrain, ytrain, xtest, ytest, tr_shape, ts_shape = \
prepare_train_test_data( False, files_list)

xtrain = np.reshape( xtrain, (tr_shape[0], tr_shape[1], -1) )
ytrain = np.reshape( ytrain, (tr_shape[0], 1) ) 
xtest  = np.reshape( xtest,  (ts_shape[0], ts_shape[1], -1) )
ytest = np.reshape(  ytest,  (ts_shape[0], 1) )

print np.shape(xtrain), np.shape(ytrain), np.shape(xtest), np.shape(ytest)


# --- plain discriminative --- 

# --- network set-up ---
in_out_neurons = 1
hidden_neurons = 256
win_size = 200

batch_size = 128


# --- prepare ---

with open("res/tsRnn.txt", "w") as text_file:
    text_file.close()

    
# --- optimizer ---
sgd  = SGD(lr = 0.01, momentum = 0.9, nesterov = True)
rms  = RMSprop(lr = 0.05,  rho = 0.9, epsilon  = 1e-08, decay = 0.0)


adam = Adam(lr = 0.002, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0)

model = Sequential()
model.add( LSTM(hidden_neurons, \
                input_dim = in_out_neurons, \
                return_sequences = False, \
                input_shape = [batch_size, win_size, in_out_neurons ], \
#               input_length = win_size, \
                activation ='tanh',\
#               dropout = 0.1,\
#               kernel_regularizer = l2(0.2), 
#               recurrent_regularizer = l2(0.1),\
#               !change!
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


#model.add(Dropout(0.7))
model.add(Dense(128,\
#                    kernel_regularizer = l2(0.01), \
                    kernel_initializer = he_normal()))
model.add(Activation("relu"))
model.add(Dense(64, \
#                    kernel_regularizer = l2(0.01), \
                    kernel_initializer = he_normal()))
model.add(Activation("relu"))
model.add(Dense(32, \
#                    kernel_regularizer = l2(0.01), \
                    kernel_initializer = he_normal()))
model.add(Activation("relu"))
model.add(Dense(1,  \
#                    kernel_regularizer = l2(0.01), \
                    kernel_initializer = he_normal()))
model.add(Activation("linear"))


filepath="res/model/plain_weights-{epoch:02d}.hdf5"
checkpointer = ModelCheckpoint(filepath, verbose=0, save_best_only=False, save_weights_only=False,\
                               period = 10 )


# training
model.compile(loss = "mean_squared_error", optimizer = adam)

model.fit( xtrain, ytrain, shuffle = True, \
           callbacks = [ \
           TestCallback( (xtest, ytest), (xtrain, ytrain) ), checkpointer ], \
           batch_size = batch_size, epochs = 500)

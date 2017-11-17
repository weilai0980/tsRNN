
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

#import sys
#sys.path.append("/usr/local/lib/python2.7/dist-packages/tensorflow/models/tutorials/rnn/translate")


# To Do:
# context based generative
# time stamp as input features 

# TO DO
# model differenced data 
# trend 


# --- results --- 
res_file = "res/tsRnn_gen.txt"

# --- set-up ---
hidden_neurons = 64
para_lr = 0.00001
para_batch_size = 512
para_epoch = 500



# validation on each epoch 
class TestCallback_Generative(Callback):
    def __init__(self, test_data, train_data):
        self.test_data  = test_data
        self.train_data = train_data

    def on_epoch_end(self, epoch, logs={}):
        x_ts, y_ts = self.test_data
        x_tr, y_tr = self.train_data
        
#         loss = self.model.evaluate(x, y, verbose=0)
        
        py_ts = self.model.predict(x_ts, verbose=0)
        py_tr = self.model.predict(x_tr, verbose=0)
        
        cnt_ts = len(x_ts)
        cnt_tr = len(x_tr)
        
        mse_ts = mean([(py_ts[i][-1][0] - y_ts[i][-1][0])**2 for i in range(cnt_ts)])
        mse_tr = mean([(py_tr[i][-1][0] - y_tr[i][-1][0])**2 for i in range(cnt_tr)])
                
        with open(res_file, "a") as text_file:
            text_file.write("At epoch %d: train %f, test %f\n" % ( epoch, sqrt(mse_tr),\
                                                                           sqrt(mse_ts) )) 
            

# --- prepare ---

with open(res_file, "w") as text_file:
    text_file.close()
            
        
# # air-quality data

files_list=["../../dataset/dataset_ts/air_xtrain.dat", \
            "../../dataset/dataset_ts/air_xtest.dat",\
            "../../dataset/dataset_ts/air_ytrain.dat", \
            "../../dataset/dataset_ts/air_ytest.dat"]




xtrain, ytrain, xtest, ytest, tr_shape, ts_shape = \
prepare_train_test_data( False, files_list)

#xtrain = np.reshape( xtrain, (tr_shape[0], tr_shape[1], -1) )
#ytrain = np.reshape( ytrain, (tr_shape[0], 1) ) 
#xtest  = np.reshape( xtest,  (ts_shape[0], ts_shape[1], -1) )
#ytest = np.reshape(  ytest,  (ts_shape[0], 1) )

print np.shape(xtrain), np.shape(ytrain), np.shape(xtest), np.shape(ytest)

if len(tr_shape)==2:
    
    gen_xtrain = np.expand_dims( xtrain, 2 )
    gen_xtest  = np.expand_dims( xtest,  2 )
    
elif len(tr_shape)==3:
    gen_xtrain = np.reshape( xtrain, tr_shape )
    gen_xtest  = np.reshape( xtest,  ts_shape )


# prepare data
ytrain = expand_y( gen_xtrain, ytrain )
ytest  = expand_y( gen_xtest , ytest  )

gen_ytrain = np.expand_dims( ytrain, 2 ) 
gen_ytest  = np.expand_dims( ytest,  2 )
    
print np.shape(gen_xtrain), np.shape(gen_ytrain), np.shape(gen_xtest), np.shape(gen_ytest)



# Generative model  

in_out_neurons = np.shape(gen_xtrain)[-1]
win_size = np.shape(gen_xtrain)[1]  
    
# optimizer 
sgd  = SGD(lr = 0.0001, momentum = 0.9, nesterov = True)
rmsp = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
adam = Adam(lr = para_lr, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0)

# network structure 
model = Sequential()
model.add( LSTM(hidden_neurons, input_dim = in_out_neurons, return_sequences = True, \
                input_length = win_size,\
                input_shape = [para_batch_size, win_size, in_out_neurons ],\
                activation   = 'sigmoid',\
#                dropout = 0.1,\
#                kernel_regularizer = l2(0.15), \
#                recurrent_regularizer = l2(0.15), \
                kernel_initializer    = glorot_normal(), \
                recurrent_initializer = glorot_normal() ) )


model.add( LSTM(64, return_sequences = True, \
#                input_length = win_size,\
                activation   = 'sigmoid',\
#                dropout = 0.1,\
#                kernel_regularizer = l2(0.15), \
#                recurrent_regularizer = l2(0.15), \
                kernel_initializer    = glorot_normal(), \
                recurrent_initializer = glorot_normal() ) )
'''
model.add( LSTM(32, input_dim = in_out_neurons, return_sequences = True, \
                input_length = win_size,\
                activation   = 'sigmoid',\
#                dropout = 0.1,\
#                kernel_regularizer = l2(0.15), \
#                recurrent_regularizer = l2(0.15), \
                kernel_initializer    = glorot_normal(), \
                recurrent_initializer = glorot_normal() ) )
'''

# identity ini
# he's ini:   he_normal()
# xavier ini: glorot_normal()
# orthogonal: Orthogonal() 


#model.add(Dropout(0.7))
#model.add(Dense(128,\
#                    kernel_regularizer = l2(0.01), \
#                    kernel_initializer = he_normal()))
#model.add(Activation("relu"))
'''
model.add(TimeDistributed(Dense(128, activation = 'relu',\
                                    kernel_regularizer = l2(0.01), \
                                    kernel_initializer = he_normal() ) ))
'''
# model.add(Dropout(0.1))
model.add(TimeDistributed(Dense(128, activation = 'relu',\
#                                    kernel_regularizer = l2(0.001), \
                                    kernel_initializer = he_normal() ) ))

model.add(TimeDistributed(Dense(64, activation = 'relu',\
#                                    kernel_regularizer = l2(0.001), \
                                    kernel_initializer = he_normal() ) ))

model.add(TimeDistributed(Dense(32, activation = 'linear',\
#                                    kernel_regularizer = l2(0.001), \
                                    kernel_initializer = he_normal() ) ))

model.add(TimeDistributed(Dense(1,  activation = 'linear',\
#                                     kernel_regularizer = l2(0.01), \
                                    kernel_initializer = he_normal() ) ))


# save model 
filepath="res/model/generative_weights-{epoch:02d}.hdf5"
checkpointer = ModelCheckpoint(filepath, verbose=0, save_best_only=False, save_weights_only=False,\
                               period = 10 )

# training
model.compile( loss = "mean_squared_error", optimizer = sgd )


model.fit( gen_xtrain, gen_ytrain, shuffle=True,  \
           callbacks = [ TestCallback_Generative( \
                         (gen_xtest, gen_ytest), (gen_xtrain, gen_ytrain) ) ], \
                         batch_size = para_batch_size, epochs = para_epoch )
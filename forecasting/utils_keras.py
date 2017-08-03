
# data processing packages
import numpy as np   
import pandas as pd 

from scipy import stats # look at scipy
from scipy import linalg
from scipy import *
 
import random

# machine leanring packages

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from keras.models import Sequential 
from keras.layers.core import Dense, Activation, Dropout, Reshape, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.optimizers import *
from keras.callbacks import *
# RMSprop, Adadelta
from keras.regularizers import *
from keras.initializers import *
from keras.activations import *


# Utilities 

# argu: np.matrix
# return: np.matrix
def normalize_y( y_mat, m, v ):
    
    tmpy = y_mat.reshape( (len(y_mat),) )
    tmpy = list(tmpy)
    
    resy = []
    
    for i in tmpy:
        resy.append( (i - m)*1.0/sqrt(v + 0.001) )
    
    return np.array(resy)        


def expand_x_local( local_size, data):
    
    list_data = list(data)
    cnt = len(list_data)
    steps = len(list_data[0])
    
    tmp_dta = []
    
    for i in range(cnt):
        tmp_dta.append([])
        for j in range(local_size-1,steps):
            tmp_dta[-1].append( list_data[i][j-local_size+1:j+1] )
    
    return np.array(tmp_dta)

# expand y, y_t -> y_1,...y_t
# argu: np.matrix
# return: np.matrix
def expand_y( x, y ):
    cnt = len(x)
    expand_y = []
    
    tmpx = list(x)
    tmpy = list(y)
    
    for i in range(cnt):
        tmp = tmpx[i][1:]
        tmp = np.append( tmp, tmpy[i] )
        
        expand_y.append( tmp )
    
    return np.array( expand_y )


def expand_x_trend( x ):
    
    tmp_x = list(x)
    cnt  = len(x)
    colcnt = len(x[0])
    
    res_x = []
    for i in range(cnt):
        res_x.append([])
        for j in range(1,colcnt):
            res_x[-1].append( (tmp_x[i][j], tmp_x[i][j] - tmp_x[i][j-1]) )
            
    return np.array(res_x)


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
        
        print('\n Training: {}, testing: {}\n'.format( sqrt(mse_tr), sqrt(mse_ts) ))
        
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
        
        print('\nTesting loss: {}, test_err:{}, train_err:{} \n'.format(\
               loss, sqrt( mean(err_ts) ), sqrt( mean(err_tr) ) ))


# validation on each epoch with normalized Y
class TestCallback_normY(Callback):
    
    def __init__(self, test_data, test_y, m_y, v_y):
        self.test_data = test_data
        
        self.orig_y = test_y
        
        self.m_y = m_y
        self.std_y = sqrt(v_y+0.001)
        

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss = self.model.evaluate(x, y, verbose=0)
        
        cnt = len(x)
        py  = self.model.predict(x, verbose=0)        
        
        mse = mean([(py[i][0]*self.std_y + self.m_y - self.orig_y[i][0])**2 \
                    for i in range(cnt)])
        
        print('\nTesting loss: {}, acc: {} \n'.format(\
               loss, sqrt(mse) ))
        



'''
# discriminative with non-overlapping local data

#  network set-up
local_size = 5
in_out_neurons = local_size
hidden_neurons = 256
win_size = 200/5

# validation on each epoch 
class TestCallback_insight(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, mse = self.model.evaluate(x, y, verbose=0)
        
        py = self.model.predict(x, verbose=0)
        cnt = len(y)
        err = [ (y[i][0]-py[i][0])**2 for i in range(cnt)]
        
        print('\nTesting loss: {}, acc: {}, mean: {} \n'.format(\
               loss, sqrt(mse), mean(err) ))

def expand_x_local_non_overlapping( local_size, list_data):
    
    cnt = len(list_data)
    steps = len(list_data[0])
    tmp_dta = []
    
    for i in range(cnt):
        tmp_dta.append([])
        for j in range( local_size-1, steps, local_size ):
            tmp_dta[-1].append( list_data[i][ j-local_size+1:j+1 ] )
    
    return tmp_dta


disc_xtrain = np.array( expand_x_local_non_overlapping( local_size, list(xtrain) ) )
disc_xtest  = np.array( expand_x_local_non_overlapping( local_size, list(xtest) ) )

disc_ytrain = ytrain
disc_ytest  = ytest

print np.shape(disc_xtrain), np.shape(disc_ytrain), np.shape(disc_xtest), \
np.shape(disc_ytest)

# optimizer 
adam = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0)

model = Sequential()
model.add(LSTM(hidden_neurons, input_dim = in_out_neurons, return_sequences = False, \
               input_length = win_size, \
#                activation = 'tanh', \
#                dropout = 0.1,\
#                kernel_regularizer = l2(0.1),      recurrent_regularizer = l2(0.1), \
               kernel_initializer = glorot_normal(), \
               recurrent_initializer = glorot_normal() ))
# identity ini
# he's ini:   he_normal()
# xavier ini: glorot_normal()
# orthogonal: Orthogonal() 

model.add(Dropout(0.1))

model.add(Dense(128, kernel_regularizer = l2(0.1), \
                    kernel_initializer = he_normal()))
model.add(Activation("relu"))
model.add(Dense(64, kernel_regularizer = l2(0.1), \
                    kernel_initializer = he_normal()))
model.add(Activation("relu"))
model.add(Dense(1,  kernel_regularizer = l2(0.1), \
                    kernel_initializer = he_normal()))
model.add(Activation("linear"))


# training 
model.compile(loss="mean_squared_error", optimizer = adam, metrics = ['mse'])


model.fit( disc_xtrain, disc_ytrain, shuffle=True, \
           callbacks = [ TestCallback_insight((disc_xtest,  disc_ytest),\
                                              (disc_xtrain, disc_ytrain)) ], \
          batch_size = 256, epochs = 500)

        
    
# discriminative with normalized Y 
        
# prepare data     
ytr_mean = mean(ytrain) 
ytr_var  = var(ytrain) 
        
disc_xtrain = np.reshape( xtrain, [-1, 100, 1] )
disc_xtest  = np.reshape( xtest,  [-1, 100, 1] )

disc_ytrain = np.reshape( normalize_y( ytrain, ytr_mean, ytr_var ), [-1,1] )
disc_ytest  = np.reshape( normalize_y( ytest,  ytr_mean, ytr_var ), [-1,1] )

print np.shape(disc_xtrain), np.shape(disc_ytrain), np.shape(disc_xtest), \
np.shape(disc_ytest)


#  network set-up
in_out_neurons = 1
hidden_neurons = 256
win_size = 100

# optimizer 
sgd  = SGD( lr = 0.01, momentum = 0.9, nesterov = True)
rms  = RMSprop(lr=0.05, rho=0.9, epsilon=1e-08, decay=0.0)

adam = Adam(lr = 0.05, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0)


model = Sequential()

model.add(LSTM(hidden_neurons, input_dim = in_out_neurons, return_sequences = False, \
               input_length = win_size, \
#                dropout = 0.1,\
               kernel_regularizer = l2(0.1),      recurrent_regularizer = l2(0.1), \
               kernel_initializer = glorot_normal(), \
               recurrent_initializer = glorot_normal() ))
# change: activiation

# identity ini
# he's ini:   he_normal()
# xavier ini: glorot_normal()
# orthogonal: Orthogonal() 


model.add(Dense(64, kernel_regularizer = l2(0.1), \
                    kernel_initializer = he_normal()))
model.add(Activation("relu"))
model.add(Dense(32, kernel_regularizer = l2(0.1), \
                    kernel_initializer = he_normal()))
model.add(Activation("relu"))
model.add(Dense(1,  kernel_regularizer = l2(0.1), \
                    kernel_initializer = he_normal()))
model.add(Activation("linear"))


# training 
model.compile(loss="mean_squared_error", optimizer = adam )


model.fit( disc_xtrain, disc_ytrain, shuffle=True, \
           callbacks = [ TestCallback_normY((disc_xtest, disc_ytest), ytest, \
                                       ytr_mean,\
                                       ytr_var ) ], \
           batch_size = 256, epochs = 500)



# discriminative with overlapping local data

#  network set-up
local_size = 10
in_out_neurons = local_size
hidden_neurons = 256

orig_win_size = win_size
win_size = 200 - local_size + 1


disc_xtrain = expand_x_local( local_size, xtrain )
disc_xtest  = expand_x_local( local_size, xtest )

disc_ytrain = ytrain
disc_ytest  = ytest

print np.shape(disc_xtrain), np.shape(disc_ytrain), np.shape(disc_xtest), \
np.shape(disc_ytest)


# optimizer 

adam = Adam(lr = 0.02, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0)

model = Sequential()
model.add(LSTM(hidden_neurons, input_dim = in_out_neurons, return_sequences = False, \
               input_length = win_size, \
               activation = 'tanh', \
               dropout = 0.1,\
               kernel_regularizer = l2(0.1),      recurrent_regularizer = l2(0.1), \
               kernel_initializer = glorot_normal(), \
               recurrent_initializer = glorot_normal() ))
# identity ini
# he's ini:   he_normal()
# xavier ini: glorot_normal()
# orthogonal: Orthogonal() 


model.add(Dense(128, kernel_regularizer = l2(0.1), \
                    kernel_initializer = he_normal()))
model.add(Activation("relu"))
model.add(Dense(32, kernel_regularizer = l2(0.1), \
                    kernel_initializer = he_normal()))
model.add(Activation("relu"))
model.add(Dense(1,  kernel_regularizer = l2(0.1), \
                    kernel_initializer = he_normal()))
model.add(Activation("linear"))


# training 
model.compile(loss="mean_squared_error", optimizer = adam, metrics = ['mse'])


model.fit( disc_xtrain, disc_ytrain, shuffle=True, \
           callbacks = [ TestCallback((disc_xtest, disc_ytest)) ], \
          batch_size = 256, epochs = 500)

'''
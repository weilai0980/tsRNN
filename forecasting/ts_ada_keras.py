
# data processing packages
import numpy as np   
import pandas as pd
from scipy import *

import random

# machine leanring packages

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from keras.models import Sequential 
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.optimizers import *
from keras.callbacks import *

from keras.regularizers import *
from keras.initializers import *
from keras.activations import *

from keras.models import Model
from keras.layers import *
from keras.layers.merge import *
from keras import backend as K

from utils_keras import *
from utils_dataPrepro import *

#--- PARAMETERS ---
N_DIM = 1
N_RNN = 128

LR = 0.003

WIN_SIZE = 200

MODEL_CHECK_PERIOD = 10
BATCH_SIZE = 64


#--- associated functions ---

def split_raw_diff(x):
    cnt = len( list(x) )
    steps = len(x[0])
    
    raw  = []
    diff = []
    last = []
    
    for i in xrange(cnt):
        raw.append( [j[0] for j in x[i]] )
        last.append( x[i][steps-1][0] )
        diff.append( [j[1] for j in x[i]] )

    return np.asarray(raw), np.asarray(diff), np.asarray(last)

# validation on each epoch 
class TestCallback_ada(Callback):
    def __init__(self, test_data, train_data):
        
        self.test_data  = test_data        
        self.train_data = train_data

    def on_epoch_end(self, epoch, logs={}):
        
        raw_ts, diff_ts, last_ts, y_ts = self.test_data
        raw_tr, diff_tr, last_tr, y_tr = self.train_data
        
        py_tr = self.model.predict( [raw_tr, diff_tr, last_tr], verbose=0)
        py_ts = self.model.predict( [raw_ts, diff_ts, last_ts], verbose=0)
        
        size_tr = len(y_tr)
        size_ts = len(y_ts)
        
        err_tr = [ (y_tr[i][0] - py_tr[i][0])**2 for i in range(size_tr) ]
        err_ts = [ (y_ts[i][0] - py_ts[i][0])**2 for i in range(size_ts) ]
        
        loss = self.model.evaluate([raw_ts, diff_ts, last_ts], y_ts, verbose=0)
        
        with open("res/tsRnn_ada.txt", "a") as text_file:
            text_file.write("At epoch %d: loss %f, train %f, test %f\n" % ( epoch, loss, sqrt(mean(err_tr)),\
                                                                           sqrt(mean(err_ts))) )

#--- load data ---

files_list=["../../dataset/dataset_ts/air_xtrain.dat", \
            "../../dataset/dataset_ts/air_xtest.dat",\
            "../../dataset/dataset_ts/air_ytrain.dat", \
            "../../dataset/dataset_ts/air_ytest.dat"]

xtrain, ytrain, xtest, ytest, tr_shape, ts_shape = \
prepare_train_test_data( True, files_list)


xtrain = np.reshape( xtrain, (tr_shape[0], tr_shape[1]-1, -1) )
ytrain = np.reshape( ytrain, (tr_shape[0], 1) ) 
xtest  = np.reshape( xtest,  (ts_shape[0], ts_shape[1]-1, -1) )
ytest = np.reshape(  ytest,  (ts_shape[0], 1) )

print "Original dataset shape:", np.shape(xtrain), np.shape(ytrain), np.shape(xtest), np.shape(ytest)

#--- tailor data ---

'''
raw_tr, diff_tr, last_tr = split_raw_diff(xtrain)
raw_ts, diff_ts, last_ts = split_raw_diff(xtest)

raw_tr  = np.reshape( raw_tr,  (tr_shape[0], tr_shape[1]-1,-1) )
diff_tr = np.reshape( diff_tr, (tr_shape[0], tr_shape[1]-1,-1) )
last_tr = np.reshape( last_tr, (tr_shape[0],-1 ) )

raw_ts  = np.reshape( raw_ts,  (ts_shape[0], ts_shape[1]-1,-1) )
diff_ts = np.reshape( diff_ts, (ts_shape[0], ts_shape[1]-1,-1) )
last_ts = np.reshape( last_ts, (ts_shape[0],-1 ) )

ytrain = np.reshape( ytrain, (tr_shape[0], 1) ) 
ytest = np.reshape(  ytest,  (ts_shape[0], 1) )

print "Training and testing dataset shape:",
print np.shape(raw_tr), np.shape(diff_tr), np.shape(last_tr)
print np.shape(raw_ts), np.shape(diff_ts), np.shape(last_ts)
'''

# --- Prepare ----

with open("res/tsRnn_ada.txt", "w") as text_file:
    text_file.close()
        
# --- build the network ---    

def difference(pair_of_tensors):
    x, y = pair_of_tensors
    return x - y

def one_minus(tensors):
    return Add()( [tensors[0], tensors[1]] )
#    return tensors[1]-tensors[0]



#with tf.device('/gpu:1'):
    
    #input_raw  = Input(shape = ( WIN_SIZE -1, 1), dtype='float32')
input_diff = Input(shape = ( WIN_SIZE -1, 2), dtype='float32')
# smoothing last values
#    input_last = Input(shape = ( 1, ), dtype='float32')

#hidden_raw  = LSTM(N_RNN)( input_raw )
#output_raw  = Dense(1, activation='relu')(hidden_raw)

hidden_diff = LSTM(N_RNN)( input_diff )

output_diff = Dense(64, activation='relu')(hidden_diff)
output_diff = Dense(1, activation='linear')(output_diff)
    
output_main = output_diff
    
    # --- training ---
    #input_raw,, input_last
model = Model(inputs=[input_diff], outputs= output_main )
adam = Adam(lr = LR , beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0)

model.compile(optimizer = adam, loss='mean_squared_error')

#    filepath="res/model/ada_weights-{epoch:02d}.hdf5"
#    checkpointer = ModelCheckpoint(filepath, verbose=0, save_best_only=False, save_weights_only=False,\
#                               period=MODEL_CHECK_PERIOD)

#    , diff_tr, last_tr raw_tr
model.fit([xtrain], [ytrain], epochs = 500, batch_size = BATCH_SIZE,\
          #callbacks = \
          #[TestCallback_ada( [ raw_tr, diff_tr, last_tr, ytrain ], [ raw_ts, diff_ts, last_ts, ytest ] ),\ 
          # checkpointer ],\
           verbose=0)
                  

'''
output_diff = Add()( [output_diff, input_last] )

weight_var = Lambda(difference)([output_diff, output_raw])

diff_prob = Dense(1, activation='sigmoid')( weight_var )
weighted_diff = merge([output_diff, diff_prob], mode='mul')
'''

'''
print "++++ shape for testing:\n", output_diff.shape, diff_prob.shape


raw_prob = Lambda(one_minus)([diff_prob, -1.0*K.ones([1,])])

print "-------- shape for testing:\n", diff_prob.shape, output_raw.shape, raw_prob.shape


# test
weighted_raw  = merge([output_raw, diff_prob], mode='mul')

#output_main = Add()( [weighted_diff, weighted_raw] )
'''
        
        
# # load weights
# model.load_weights("weights.best.hdf5")
# # Compile model (required to make predictions)
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#!/usr/bin/python

from keras.layers import LSTM, Lambda
from keras.layers import *
from keras.layers import Input
from keras.models import Model

from utils_libs import *

import sys
    
class ts_resLSTM_discriminative():
    
    def __init__(self, n_lstm_dim, n_steps, n_data_dim, n_lstm_layers,\
                 lr, l2, bool_is_stateful, n_batch_size, is_shuffle, model_file, log_file  ):
        
        self.LEARNING_RATE = lr
        self.L2 =  l2
        
        self.N_LSTM_LAYERS = n_lstm_layers
        self.N_LSTM_DIM    = n_lstm_dim
        
        self.N_STEPS    = n_steps
        self.N_DATA_DIM = n_data_dim        
        
        self.is_state_full = bool_is_stateful
        self.n_batch_size = n_batch_size
        
        self.is_shuffle = is_shuffle
        
        self.model_file = model_file
        self.log_file   = log_file 
        
        #optimizer
        self.sgd  = SGD(lr = para_lr, momentum = 0.9, nesterov = True)
        self.rms  = RMSprop(lr = 0.05,  rho = 0.9, epsilon  = 1e-08, decay = 0.0)
        self.adam = Adam(lr = self.LEARNING_RATE, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0)
        
    
    """
    The intermediate LSTM layers return sequences, while the last returns a single element.
    The input is also a sequence. In order to match the shape of input and output of the LSTM
    to sum them we can do it only for all layers but the last.
    """
    def residual_lstm_layers(self, input_x, n_data_dim, n_lstm_dim, n_layers, dropout):
                                            
        x = input_x
        
        for i in range(n_layers):
            return_sequences = i < n_layers - 1
            
            #?
            x_rnn = LSTM( n_lstm_dim, recurrent_dropout = dropout, dropout = dropout, return_sequences = return_sequences,\
                         kernel_initializer = glorot_normal(), recurrent_initializer = glorot_normal())(x)
            
            if return_sequences:
                if i > 0 or x.shape[-1] == n_lstm_dim:
                    x = Add()([x, x_rnn])
                else:
                    x = x_rnn
                # Note that the input size and RNN output has to match, due to the sum operation.
                # If we want different rnn_width, we'd have to perform the sum from layer 2 on.
            else:
                def slice_last(x):
                    return x[..., -1, :]
                
                x = Add()([Lambda(slice_last)(x), x_rnn])
            # Last layer does not return sequences, just the last element
            # so we select only the last element of the previous output.
        return x
    
    def mlp_layers(self, input_x, n_layers):
        
        #?
        x = Dropout(rate = 0.1)(input_x)
        
        x = Dense(32, activation='relu', kernel_initializer = he_normal() )(x)
        x = Dense(16,  activation='relu', kernel_initializer = he_normal())(x)
        #?
        #x = Dense(64,  activation='linear', kernel_initializer = he_normal())(x)
        x = Dense(1,   activation='linear', kernel_initializer = he_normal())(x)
        #regularizers.l2(0.01)
        return x
    
    
    def model_fit(self, x_tr, y_tr, x_ts, y_ts):
        
        # Stateless-validation on each epoch
        class TestCallback_stateless(Callback):
            def __init__(self, test_data, train_data, log_file):
                self.test_data  = test_data
                self.train_data = train_data
                self.log_file   = log_file
                
            def on_epoch_end(self, epoch, logs={}):
                x_ts, y_ts = self.test_data
                x_tr, y_tr = self.train_data
        
                py_tr = self.model.predict(x_tr, verbose=0)
                py_ts = self.model.predict(x_ts, verbose=0)
        
                #training and testing errors 
                err_tr = [ (i[1] - py_tr[i[0]][0])**2 for i in enumerate(y_tr) ]
                err_ts = [ (i[1] - py_ts[i[0]][0])**2 for i in enumerate(y_ts) ]
        
                #loss
                loss = self.model.evaluate(x_tr, y_tr, verbose=0)
                with open(self.log_file, "a") as text_file:
                    text_file.write("At epoch %d: loss %f, train %f, test %f\n" % ( epoch, loss, sqrt(mean(err_tr)),\
                                                                           sqrt(mean(err_ts))) )  
        
        
        # network structure 
        input_x = Input( (self.N_STEPS, self.N_DATA_DIM) )
        
        rnn_output = self.residual_lstm_layers(input_x, self.N_DATA_DIM, self.N_LSTM_DIM, self.N_LSTM_LAYERS, 0.0)        
        py_tr      = self.mlp_layers(rnn_output, 4)
        
        model = Model(inputs = input_x, outputs = py_tr)
        model.compile(optimizer = self.adam, loss = 'mean_squared_error')
        
        model.summary()
        
        # model saver
        checkpointer = ModelCheckpoint(self.model_file, verbose=0, save_best_only=False, save_weights_only=False,\
                               period = 10 )
        #begin training
        model.fit( x = x_tr, y = y_tr, shuffle = self.is_shuffle, \
           callbacks = [ \
           TestCallback_stateless( (x_ts, y_ts), (x_tr, y_tr), self.log_file ), checkpointer ], \
           batch_size = self.n_batch_size, epochs = 500)

    
if __name__ == '__main__':
    
    dataset_str = str(sys.argv[1])
    
    file_dic = {}
    file_addr = ["../../dataset/dataset_ts/air_xtrain.dat", \
            "../../dataset/dataset_ts/air_xtest.dat",\
            "../../dataset/dataset_ts/air_ytrain.dat", \
            "../../dataset/dataset_ts/air_ytest.dat"]
    file_dic.update( {"air": file_addr} )
     
    print "Loading file at", file_dic[dataset_str][0] 
    files_list = file_dic[dataset_str]
    
# --- load data and prepare data --- 
    xtrain, ytrain, xtest, ytest, tr_shape, ts_shape = prepare_train_test_data(False, files_list)
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

    # parameters set via data 
    para_input_dim = np.shape(xtrain)[-1]
    para_win_size = np.shape(xtrain)[1]
    
# --- training logs --- 
    log_file   = "res/tsRnn_discr_resnet.txt"
    model_file = "res/model/tsRnn_discr_resnet-{epoch:02d}.hdf5"

    #clean logs
    with open(log_file, "w") as text_file:
        text_file.close()

# --- network set-up ---
    #?
    para_hidden_neurons = 64
    para_layers = 3

    #?
    para_lr = 0.001
    para_batch_size = 128

    para_is_stateful = False
    para_epochs = 500
    para_shuffle = True

#--- begin training ---
    
    resnet = ts_resLSTM_discriminative( para_hidden_neurons, para_win_size, para_input_dim, para_layers,\
                 para_lr, 0.0, para_is_stateful, para_batch_size, para_shuffle, model_file, log_file )
    
    resnet.model_fit( xtrain, ytrain, xtest, ytest )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
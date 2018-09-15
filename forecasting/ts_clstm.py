#!/usr/bin/python

import sys

import collections
import hashlib
import numbers

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.rnn_cell_impl import *

# local 
from mv_rnn_cell import *
from utils_libs import *
from ts_mv_rnn_attention import *

'''
reference: Neural Granger Causality for Nonlinear Time Series
           An Interpretable and Sparse Neural Network Model for Nonlinear Granger Causality Discovery
'''

# ---- cLSTM ----

class cLSTM_causal():
    
    def __init__(self, n_dense_dim_layers, n_lstm_dim_layers, n_steps, n_data_dim, session,\
                 lr, l2_dense, max_norm , n_batch_size, bool_residual, att_type, l2_attm, l2_lasso):
        
        self.LEARNING_RATE = lr
        self.L2 =  l2
        
        self.n_lstm_dim_layers = n_lstm_dim_layers
        
        self.N_STEPS    = n_steps
        self.N_DATA_DIM = n_data_dim
        
        self.n_dense_dim_layers = n_dense_dim_layers
        self.n_batch_size       = n_batch_size
        
        self.att_type = att_type
        
        # placeholders
        self.x = tf.placeholder(tf.float32, [None, self.N_STEPS, self.N_DATA_DIM])
        self.y = tf.placeholder(tf.float32, [None, 1])
        self.keep_prob = tf.placeholder(tf.float32, [None])
        
        # begin to build the graph
        self.sess = session
        
        # feed into LSTM
        h, _ = plain_lstm( self.x, n_lstm_dim_layers, 'lstm', self.keep_prob )
        
        print(' --- cLSTM RNN using no attention:  ')
            
        # obtain the last hidden state
        tmp_hiddens = tf.transpose( h, [1,0,2] )
        h = tmp_hiddens[-1]
            
        # dropout
        h, regu_dense = plain_dense( h, n_lstm_dim_layers[-1], n_dense_dim_layers, 'dense', \
                                        tf.gather(self.keep_prob, 0), max_norm )
        #?
        self.regularization = l2_dense*regu_dense
            
        #dropout
        h = tf.nn.dropout(h, tf.gather(self.keep_prob, 1))
        
        with tf.variable_scope("output"):
            
            w = tf.get_variable('w', shape=[n_dense_dim_layers[-1], 1],\
                                     initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.zeros( [ 1 ] ))
            
            self.py = tf.matmul(h, w) + b
            
            # regularization
            # ?
            self.regularization += l2_dense*tf.nn.l2_loss(w)
            
        # ---- causal regularization
        
        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        
        # self._kernel = self.add_variable(_WEIGHTS_VARIABLE_NAME, shape=[input_depth + h_depth, 4 * self._num_units],
        
        # lstm_matrix = math_ops.matmul( array_ops.concat([inputs, m_prev], 1), self._kernel)
        
        # lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)
        
        # i, j, f, o = array_ops.split(value=lstm_matrix, num_or_size_splits=4, axis=1)
        
        # 'lstm/rnn/lstm_cell/kernel:0', 'lstm/rnn/lstm_cell/bias:0',
        
        # tf.slice(t, [1, 0, 0], [1, 1, 3])
        #[ tf_var.eval() for tf_var in tf.trainable_variables() if (vari_keyword in tf_var.name) ]
        
        
        # extract input-hidden transition weight in LSTM
        for tf_var in tf.trainable_variables():
            if tf_var.name == 'lstm/rnn/lstm_cell/kernel:0':
                lstm_kernel = tf_var
        
        self.input_hidden = tf.slice( lstm_kernel, [0, 0], [self.N_DATA_DIM, 4*n_lstm_dim_layers[-1]] )
        
        # group lasso for causal selection
        group_lasso = tf.reduce_sum( tf.sqrt(tf.reduce_sum(tf.square(self.input_hidden), 1)) )
        self.regularization += (l2_lasso*group_lasso) 
        
        
    def train_ini(self):
        
        # loss function 
        self.error_mse = tf.reduce_mean( tf.square(self.y - self.py) )
        self.error_sqsum  = tf.reduce_sum( tf.square(self.y - self.py) )
        
        # ?
        self.loss = self.error_mse + self.regularization
        
        # optimizer 
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.loss)  
        
        # initilization 
        self.init = tf.global_variables_initializer()
        self.sess.run( [self.init] )
        
    def train_batch(self, x_batch, y_batch, keep_prob ):
        
        _, c, err_sum = self.sess.run([self.optimizer, self.loss, self.error_sqsum ],\
                                      feed_dict = {self.x:x_batch, self.y:y_batch, self.keep_prob:keep_prob })
        return c, err_sum

#   initialize inference         
    def inference_ini(self):

        # error metric
        self.rmse = tf.sqrt( tf.reduce_mean(tf.square(self.y - self.py)) )
        self.mae =  tf.reduce_mean( tf.abs(self.y - self.py) )
        self.mape = tf.reduce_mean( tf.abs((self.y - self.py)*1.0/(self.y+1e-5)) )

#   infer givn testing data
    def inference(self, x_test, y_test, keep_prob):
        
        if self.att_type == '':
            return self.sess.run([self.py, self.rmse, self.mae, self.mape], \
                             feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
        
        else:
            return self.sess.run([self.att, self.py, self.rmse, self.mae, self.mape], \
                             feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    def predict(self, x_test, y_test, keep_prob):
        return self.sess.run([self.py], feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    def test_attention(self, x_test, y_test, keep_prob):
        return self.sess.run( self.att,  feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    def collect_weight_values(self):
        
        return tf.sqrt(tf.reduce_sum(tf.square(self.input_hidden), 1)).eval()
        #, self.input_hidden.eval()
    
    # collect the optimized variable values
    def collect_coeff_values(self, vari_keyword):

        return self.input_hidden.get_shape()
        
        #return [ [tf_var.name, tf_var.get_shape()] for tf_var in tf.trainable_variables() ]
    #if (vari_keyword in tf_var.name)
    # tf.shape(tf_var)
    #[ tf_var.eval() for tf_var in tf.trainable_variables() if (vari_keyword in tf_var.name) ]
    
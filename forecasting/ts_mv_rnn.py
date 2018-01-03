#!/usr/bin/python

import sys

import collections
import hashlib
import numbers

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.rnn_cell_impl import *

# local 
from custom_rnn_cell import *
from utils_libs import *

    
# ---- utilities for residual and plain layers ----  
    
def res_lstm(x, hidden_dim, n_layers, scope, dropout_keep_prob):
    
    #dropout
    #x = tf.nn.dropout(x, dropout_keep_prob)
    
    with tf.variable_scope(scope):
            #Deep lstm: residual or highway connections 
            lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_dim, \
                                                initializer= tf.contrib.keras.initializers.glorot_normal())
            hiddens, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = x, dtype = tf.float32)
            
    for i in range(1, n_layers):
        
        with tf.variable_scope(scope+str(i)):
            
            tmp_h = hiddens
            
            lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_dim, \
                                                    initializer= tf.contrib.keras.initializers.glorot_normal())
            hiddens, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = hiddens, dtype = tf.float32)
            hiddens = hiddens + tmp_h 
             
    return hiddens, state

def plain_lstm(x, dim_layers, scope, dropout_keep_prob):
    
    #dropout
    #x = tf.nn.dropout(x, dropout_keep_prob)
    
    with tf.variable_scope(scope):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(dim_layers[0], \
                                                initializer= tf.contrib.keras.initializers.glorot_normal())
        hiddens, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = x, dtype = tf.float32)
            
    for i in range(1,len(dim_layers)):
        with tf.variable_scope(scope+str(i)):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(dim_layers[i], \
                                                    initializer= tf.contrib.keras.initializers.glorot_normal())
            hiddens, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = hiddens, dtype = tf.float32)
                
    return hiddens, state 

    
def res_dense(x, x_dim, hidden_dim, n_layers, scope, dropout_keep_prob):
    
        #dropout
        x = tf.nn.dropout(x, dropout_keep_prob)
        
        with tf.variable_scope(scope):
                # initilization
                w = tf.get_variable('w', [x_dim, hidden_dim], dtype = tf.float32,
                                          initializer = tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros([hidden_dim]))
                h = tf.nn.relu(tf.matmul(x, w) + b )

                regularization = tf.nn.l2_loss(w)
        #dropout
        #h = tf.nn.dropout(h, dropout_keep_prob)
        
        for i in range(1, n_layers):
            
            with tf.variable_scope(scope+str(i)):
                w = tf.get_variable('w', [hidden_dim, hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros( hidden_dim ))
                
                # residual connection
                tmp_h = h
                h = tf.nn.relu( tf.matmul(h, w) + b )
                h = tmp_h + h
                
                regularization += tf.nn.l2_loss(w)
        
        return h, regularization
    
def plain_dense(x, x_dim, dim_layers, scope, dropout_keep_prob):
    
        #dropout
        x = tf.nn.dropout(x, dropout_keep_prob)
        
        with tf.variable_scope(scope):
                # initilization
                w = tf.get_variable('w', [x_dim, dim_layers[0]], dtype=tf.float32,\
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros([dim_layers[0]]))
                h = tf.nn.relu( tf.matmul(x, w) + b )

                #?
                regularization = tf.nn.l2_loss(w)
                #regularization = tf.reduce_sum(tf.abs(w))
                
        #dropout
        #h = tf.nn.dropout(h, dropout_keep_prob)
        
        for i in range(1, len(dim_layers)):
            
            with tf.variable_scope(scope+str(i)):
                w = tf.get_variable('w', [dim_layers[i-1], dim_layers[i]], dtype=tf.float32,\
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros( dim_layers[i] ))
                h = tf.nn.relu( tf.matmul(h, w) + b )
                
                #?
                regularization += tf.nn.l2_loss(w)
                #regularization += tf.reduce_sum(tf.abs(w))
                
        return h, regularization

    
#---- Attention plain ----

# ref: a structured self attentive sentence embedding  
def attention_temp_mlp( h, h_dim, att_dim, scope ):
    # tf.tensordot
    with tf.variable_scope(scope):
        
        w = tf.get_variable('w', [h_dim, att_dim], initializer=tf.contrib.layers.xavier_initializer())
        #? add bias ?
        tmp_h = tf.nn.relu( tf.tensordot(h, w, axes=1) )

        w_logit = tf.get_variable('w_log', [att_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
        logit = tf.tensordot(tmp_h, w_logit, axes=1)
        
        alphas = tf.nn.softmax( tf.squeeze(logit) )
        
    return tf.reduce_sum(h*tf.expand_dims(alphas, -1), 1), alphas

def attention_temp_logit( h, h_dim, scope ):
    # tf.tensordot
    with tf.variable_scope(scope):
        
        w = tf.get_variable('w', [h_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.zeros([1, 1]))

        #? bias and nonlinear activiation 
        logit = tf.squeeze(tf.tensordot(h, w, axes=1))
        
        alphas = tf.nn.softmax( tf.squeeze(logit) )
        
    return tf.reduce_sum(h*tf.expand_dims(alphas, -1), 1), alphas, tf.nn.l2_loss(w) 

def attention_temp_logit_concat( h, h_dim, scope, step ):
    # tf.tensordot
    
    h_context, h_last = tf.split(h, [step-1, 1], 1)
    h_last = tf.squeeze(h_last)
    
    with tf.variable_scope(scope):
        
        w = tf.get_variable('w', [h_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.zeros([1, 1]))
        
        #? bias and nonlinear activiation 
        logit = tf.squeeze(tf.tensordot(h_context, w, axes=1))
        
        alphas = tf.nn.softmax( tf.squeeze(logit) )
        
        context = tf.reduce_sum(h_context*tf.expand_dims(alphas, -1), 1)
        
    return tf.concat([context, h_last], 1), alphas, tf.nn.l2_loss(w) 
    
    
# ---- plain RNN ----

class tsLSTM_plain():
    
    def __init__(self, n_dense_dim_layers, n_lstm_dim_layers, n_steps, n_data_dim, session,\
                 lr, l2, max_norm , n_batch_size, bool_residual, bool_att):
        
        self.LEARNING_RATE = lr
        self.L2 =  l2
        
        self.n_lstm_dim_layers = n_lstm_dim_layers
        
        self.N_STEPS    = n_steps
        self.N_DATA_DIM = n_data_dim
        
        self.n_dense_dim_layers = n_dense_dim_layers
        self.n_batch_size       = n_batch_size
        
        # placeholders
        self.x = tf.placeholder(tf.float32, [None, self.N_STEPS, self.N_DATA_DIM])
        self.y = tf.placeholder(tf.float32, [None, 1])
        self.keep_prob = tf.placeholder(tf.float32)
        
        # begin to build the graph
        self.sess = session
        
        if bool_residual == True:
            
            h, _ = res_lstm( self.x, n_lstm_dim_layers[0], len(n_lstm_dim_layers), 'lstm', self.keep_prob )
            
            if bool_att == 'temp':
                h, self.att, regu = attention_temp_logit( h, n_lstm_dim_layers[0], 'att' )
                h, regul = res_dense( h, n_lstm_dim_layers[0], n_dense_dim_layers[0], len(n_dense_dim_layers), 'dense',\
                                  self.keep_prob )
                
                regu_all = regu + regul 
            
            else:
                # obtain the last hidden state    
                tmp_hiddens = tf.transpose( h, [1,0,2]  )
                h = tmp_hiddens[-1]
            
                h, regul = res_dense( h, n_lstm_dim_layers[0], n_dense_dim_layers[0], len(n_dense_dim_layers), 'dense',\
                                  self.keep_prob )
                regu_all = regul
                
        else:
            
            h, _ = plain_lstm( self.x, n_lstm_dim_layers, 'lstm', self.keep_prob )
            
            if bool_att == 'temp':
                
                h, self.att, regu = attention_temp_logit_concat( h, n_lstm_dim_layers[-1], 'att', self.N_STEPS )
                
                h, regul = plain_dense( h, n_lstm_dim_layers[-1]*2, n_dense_dim_layers, 'dense', self.keep_prob )
                
                #?
                regu_all = regu + regul 
            
            else:
                # obtain the last hidden state
                tmp_hiddens = tf.transpose( h, [1,0,2]  )
                h = tmp_hiddens[-1]
                
                h, regul = plain_dense( h, n_lstm_dim_layers[-1], n_dense_dim_layers, 'dense', self.keep_prob )
                regu_all = regul 
            
        #dropout
        #last_hidden = tf.nn.dropout(last_hidden, self.keep_prob)
        with tf.variable_scope("output"):
            
            w = tf.get_variable('w', shape=[n_dense_dim_layers[-1], 1],\
                                     initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.zeros( [ 1 ] ))
            
            self.py = tf.matmul(h, w) + b
            self.regularization = regu_all + tf.nn.l2_loss(w)
            
    def train_ini(self):  
        
        # loss function 
        self.cost = tf.reduce_mean( tf.square(self.y - self.py) ) + self.L2*self.regularization
        
        self.optimizer = \
        tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)  
#         tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)  
#         !! same lr, converge faster
#         tf.train.AdadeltaOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)
#         tf.train.RMSPropOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)
#         tf.train.RMSPropOptimizer(learning_rate = self.lr).minimize(self.cost)
#         tf.train.AdadeltaOptimizer(learning_rate = self.lr).minimize(self.cost)
#         tf.train.GradientDescentOptimizer(learning_rate = self.lr).minimize(self.cost)
        
        self.init = tf.global_variables_initializer()
        self.sess.run( [self.init] )
        
    def train_batch(self, x_batch, y_batch, keep_prob ):
        
        _, c = self.sess.run([self.optimizer, self.cost],\
                              feed_dict={self.x:x_batch, self.y:y_batch, self.keep_prob:keep_prob })
        return c

#   initialize inference         
    def inference_ini(self):

#       denormalzied RMSE  
        self.rmse = tf.sqrt( tf.reduce_mean( tf.square( self.y - self.py ) ) )
        self.y_hat= self.py
        
#   infer givn testing data    
    def inference(self, x_test, y_test, keep_prob):
        return self.sess.run([self.rmse],  feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    def predict(self, x_test, y_test, keep_prob):
        return self.sess.run([self.y_hat], feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    def test_attention(self, x_test, y_test, keep_prob):
        return self.sess.run([self.att],  feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    
    
# ---- Attention for Sep RNN ----
'''
def attention_variate_aggre_hidden(alpha, h_list, h_dim_list):
    
    bool_equ_dim = True
    for i in h_dim_list:
        if i != h_dim_list[0]:
            bool_equ_dim = False
    
    if bool_equ_dim == True:
        return tf.reduce_sum(h*tf.expand_dims(alpha, -1), 1)
    else:
        tmph = []
        for i in range(len(h_list)):
            tmph.append( h_list[i]*alpha[i] )
        
        return tf.concat(tmph, 1)
'''    
# shape of h_list: [#variate, batch_size, steps, dim]
def attention_variate_temp_mlp( h_list, h_dim, scope, step ):
    
    with tf.variable_scope(scope):
        
        w_temp = tf.get_variable('w_temp', [len(h_list), 1, 1, h_dim], initializer=tf.contrib.layers.xavier_initializer())
        b_temp = tf.Variable( tf.random_normal([len(h_list), 1, 1, 1]) )
        
        tmph = tf.stack(h_list, 0)
        tmph_before, tmph_last = tf.split(tmph, [step-1, 1], 2)
        tmph_last = tf.squeeze( tmph_last, [2] )
        
        # ? bias nonlinear activation ?
        temp_logit = tf.reduce_sum( tmph_before*w_temp, 3 )
        temp_weight = tf.nn.softmax( temp_logit )
        
        tmph_cxt = tf.reduce_sum(tmph_before*tf.expand_dims(temp_weight, -1), 2)
        h_temp = tf.concat([tmph_cxt, tmph_last], 2)
        
        w_var = tf.get_variable('w_var', [h_dim*2, 1], initializer=tf.contrib.layers.xavier_initializer())
        b_var = tf.Variable( tf.random_normal([1]) )
        # ? bias nonlinear activation ?
        var_weight = tf.sigmoid( tf.tensordot(h_temp, w_var, axes=1) + b_var )
        
        h_var_list = tf.split(h_temp*var_weight, num_or_size_splits = len(h_list), axis = 0) 
        
    return tf.squeeze(tf.concat(h_var_list, 2)), tf.nn.l2_loss(w_var) + tf.nn.l2_loss(w_temp), [temp_weight, var_weight]


def attention_temp_logit( h_list, h_dim, scope, step ):
    
    with tf.variable_scope(scope):
        
        w_temp = tf.get_variable('w_', [len(h_list), 1, 1, h_dim], initializer=tf.contrib.layers.xavier_initializer())
        b_temp = tf.Variable( tf.random_normal([len(h_list), 1, 1, 1]) )
        
        tmph = tf.stack(h_list, 0)
        tmph_before, tmph_last = tf.split(tmph, [step-1, 1], 2)
        tmph_last = tf.squeeze( tmph_last, [2] )
        
        # ? bias nonlinear activation ?
        temp_logit = tf.reduce_sum( tmph_before*w_temp, 3 )
        temp_weight = tf.nn.softmax( temp_logit )
        
        tmph_cxt = tf.reduce_sum( tmph_before*tf.expand_dims(temp_weight, -1), 2)
        
        #?
        h_temp = tf.concat([tmph_cxt, tmph_last], 2)
        #h_temp = tmph_cxt + tmph_last
        
        h_var_list = tf.split(h_temp, num_or_size_splits = len(h_list), axis = 0) 
        
    return tf.squeeze(tf.concat(h_var_list, 2)), tf.nn.l2_loss(w_temp), temp_weight

def attention_variate_temp_logit( h_list, h_dim, scope, step ):
    
    with tf.variable_scope(scope):
        
        w_temp = tf.get_variable('w_temp', [len(h_list), 1, 1, h_dim], initializer=tf.contrib.layers.xavier_initializer())
        b_temp = tf.Variable( tf.random_normal([len(h_list), 1, 1, 1]) )
        
        tmph = tf.stack(h_list, 0)
        tmph_before, tmph_last = tf.split(tmph, [step-1, 1], 2)
        tmph_last = tf.squeeze( tmph_last, [2] )
        
        # ? bias nonlinear activation ?
        temp_logit = tf.reduce_sum( tmph_before*w_temp, 3 )
        temp_weight = tf.nn.softmax( temp_logit )
        
        tmph_cxt = tf.reduce_sum(tmph_before*tf.expand_dims(temp_weight, -1), 2)
        h_temp = tf.concat([tmph_cxt, tmph_last], 2)
        
        w_var = tf.get_variable('w_var', [h_dim*2, 1], initializer=tf.contrib.layers.xavier_initializer())
        b_var = tf.Variable( tf.random_normal([1]) )
        
        # ? bias nonlinear activation ?
        var_weight = tf.sigmoid( tf.tensordot(h_temp, w_var, axes=1) + b_var )
        
        h_var_list = tf.split(h_temp*var_weight, num_or_size_splits = len(h_list), axis = 0) 
        
    return tf.squeeze(tf.concat(h_var_list, 2)), tf.nn.l2_loss(w_var) + tf.nn.l2_loss(w_temp), [temp_weight, var_weight]

        
#---- separate RNN ----

class tsLSTM_seperate():
    
    def __init__(self, n_dense_dim_layers, n_lstm_dim_layers, n_steps, n_data_dim, session,\
                 lr, l2, max_norm , n_batch_size, bool_residual, bool_att):
        
        self.LEARNING_RATE = lr
        self.L2 =  l2
        
        self.n_lstm_dim_layers = n_lstm_dim_layers
        
        self.N_STEPS    = n_steps
        self.N_DATA_DIM = n_data_dim
        
        self.x = tf.placeholder(tf.float32, [None, self.N_STEPS, self.N_DATA_DIM])
        self.y = tf.placeholder(tf.float32, [None, 1])
        
        self.keep_prob = tf.placeholder(tf.float32)
        
        self.n_dense_dim_layers = n_dense_dim_layers
        self.n_batch_size       = n_batch_size
        
        self.sess = session
        
        indivi_ts = tf.split(self.x, num_or_size_splits = self.N_DATA_DIM, axis = 2)
        concat_h  = []
        
        for i in range( self.N_DATA_DIM ):
            
            current_x = indivi_ts[i]
            
            if bool_residual == True:
                h, _  = res_lstm( current_x, n_lstm_dim_layers[0], len(n_lstm_dim_layers), 'lstm'+str(i), self.keep_prob)
            else:
                h, _  = plain_lstm( current_x, n_lstm_dim_layers, 'lstm'+str(i), self.keep_prob)
                
            if bool_att == 'both' or bool_att == 'temp':
                concat_h.append(h)
            else:
                # obtain the last hidden state    
                tmp_hiddens = tf.transpose( h, [1,0,2]  )
                h = tmp_hiddens[-1]
                
                concat_h.append(h)
        
        # no attention
        if bool_att == '': 
            
            # hidden space merge
            h = tf.concat(concat_h, 1)
            h, regul = plain_dense(h, n_lstm_dim_layers[-1]*self.N_DATA_DIM, n_dense_dim_layers, 'dense', self.keep_prob)
            
            self.regularization = regul
        
        elif bool_att == 'temp':
            
            h, att_regu, self.att = attention_temp_logit( concat_h, n_lstm_dim_layers[-1], 'attention', self.N_STEPS )
            
            # dense layers 
            h, regul = plain_dense(h, n_lstm_dim_layers[-1]*self.N_DATA_DIM*2, n_dense_dim_layers, 'dense', self.keep_prob)
            self.regularization = regul + att_regu
            
        elif bool_att == 'both':
            
            h, att_regu, self.att = attention_variate_temp_logit( concat_h, n_lstm_dim_layers[-1], 'attention',self.N_STEPS )
            
            # dense layers 
            h, regul = plain_dense(h, n_lstm_dim_layers[-1]*self.N_DATA_DIM*2, n_dense_dim_layers, 'dense', self.keep_prob)
            self.regularization = regul + att_regu
            
        
        #dropout
        #h = tf.nn.dropout(h, self.keep_prob)
        with tf.variable_scope("output"):
            w = tf.get_variable('w', shape=[n_dense_dim_layers[-1], 1],\
                                     initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.zeros([ 1 ]))
            
            self.py = tf.matmul(h, w) + b
            self.regularization += tf.nn.l2_loss(w)
            
    def train_ini(self):  
        # loss function 
        self.cost = tf.reduce_mean( tf.square(self.y - self.py) ) + self.L2 * self.regularization
        
        self.optimizer = \
        tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)  
#         tf.train.AdadeltaOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)
#         tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)  
#         !! same lr, converge faster
#         tf.train.AdadeltaOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)
#         tf.train.RMSPropOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)
#         tf.train.RMSPropOptimizer(learning_rate = self.lr).minimize(self.cost)
#         tf.train.AdadeltaOptimizer(learning_rate = self.lr).minimize(self.cost)
#         tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.cost)
#         tf.train.GradientDescentOptimizer(learning_rate = self.lr).minimize(self.cost)
        
        self.init = tf.global_variables_initializer()
        self.sess.run( [self.init] )
        
    def train_batch(self, x_batch, y_batch, keep_prob ):
        _, c = self.sess.run([self.optimizer, self.cost],feed_dict={self.x:x_batch, self.y:y_batch, self.keep_prob:keep_prob})
        return c

#   initialize inference         
    def inference_ini(self):

#       denormalzied RMSE  
        self.rmse = tf.sqrt( tf.reduce_mean( tf.square( self.y - self.py ) ) )
        
#   infer givn testing data 
    def inference(self, x_test, y_test, keep_prob):
        return self.sess.run([self.rmse], feed_dict={self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    def test_attention(self, x_test, y_test, keep_prob):
        return self.sess.run([self.att],  feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    def testfunc(self, x_batch, y_batch, keep_prob ):
        return self.sess.run([self.test],\
                              feed_dict={self.x:x_batch, self.y:y_batch, self.keep_prob:keep_prob })


    
# ---- Attention for MV-RNN ----

def mv_attention_temp_logit_all( h_list, h_dim, scope, step ):
    
    with tf.variable_scope(scope):
        
        w_temp = tf.get_variable('w_', [len(h_list), 1, 1, h_dim], initializer=tf.contrib.layers.xavier_initializer())
        b_temp = tf.Variable( tf.random_normal([len(h_list), 1, 1, 1]) )
        
        tmph = tf.stack(h_list, 0)
        #tmph_before, tmph_last = tf.split(tmph, [step-1, 1], 2)
        #tmph_last = tf.squeeze( tmph_last, [2] )
        
        # ? bias nonlinear activation ?
        temp_logit = tf.reduce_sum( tmph*w_temp+b_temp, 3 )
        temp_weight = tf.nn.softmax( temp_logit )
        
        #tmph_cxt = tf.reduce_sum( tmph_before*tf.expand_dims(temp_weight, -1), 2)
        #tf.concat([tmph_cxt, tmph_last], 2)
        h_temp = tf.reduce_sum(tmph*tf.expand_dims(temp_weight, -1), 2)
        
        h_var_list = tf.split(h_temp, num_or_size_splits = len(h_list), axis = 0) 
        
    return tf.squeeze(tf.concat(h_var_list, 2)), tf.nn.l2_loss(w_temp), temp_weight

#[hi + ht]
def mv_attention_temp_logit_add( h_list, h_dim, scope, step ):
    
    with tf.variable_scope(scope):
        
        w_temp = tf.get_variable('w_', [len(h_list), 1, 1, h_dim], initializer=tf.contrib.layers.xavier_initializer())
        b_temp = tf.Variable( tf.random_normal([len(h_list), 1, 1, 1]) )
        
        tmph = tf.stack(h_list, 0)
        tmph_before, tmph_last = tf.split(tmph, [step-1, 1], 2)
        tmph_last = tf.squeeze( tmph_last, [2] )
        
        # ? bias nonlinear activation ?
        temp_logit = tf.reduce_sum( tmph_before*w_temp+b_temp, 3 )
        temp_weight = tf.nn.softmax( temp_logit )
        
        tmph_cxt = tf.reduce_sum(tmph_before*tf.expand_dims(temp_weight, -1), 2)
        
        h_temp = tmph_cxt + tmph_last
        
        h_var_list = tf.split(h_temp, num_or_size_splits = len(h_list), axis = 0) 
        
    return tf.squeeze(tf.concat(h_var_list, 2)), tf.nn.l2_loss(w_temp), temp_weight

#[hi, ht]
def mv_attention_temp_logit_concat( h_list, h_dim, scope, step ):
    
    with tf.variable_scope(scope):
        
        w_temp = tf.get_variable('w_temp', [len(h_list), 1, 1, h_dim], initializer=tf.contrib.layers.xavier_initializer())
        b_temp = tf.Variable( tf.random_normal([len(h_list), 1, 1, 1]) )
        
        tmph = tf.stack(h_list, 0)
        tmph_before, tmph_last = tf.split(tmph, [step-1, 1], 2)
        tmph_last = tf.squeeze( tmph_last, [2] )
        
        # ? bias nonlinear activation ?
        temp_logit = tf.reduce_sum( tmph_before*w_temp, 3 )
        temp_weight = tf.nn.softmax( temp_logit )
        
        tmph_cxt = tf.reduce_sum(tmph_before*tf.expand_dims(temp_weight, -1), 2)
        h_temp = tf.concat([tmph_cxt, tmph_last], 2)
        
        h_var_list = tf.split(h_temp, num_or_size_splits = len(h_list), axis = 0) 
        
    return tf.squeeze(tf.concat(h_var_list, 2)), tf.nn.l2_loss(w_temp), temp_weight

def mv_attention_variate_temp_logit( h_list, h_dim, scope, step ):
    
    with tf.variable_scope(scope):
        
        w_temp = tf.get_variable('w_temp', [len(h_list), 1, 1, h_dim], initializer=tf.contrib.layers.xavier_initializer())
        b_temp = tf.Variable( tf.random_normal([len(h_list), 1, 1, 1]) )
        
        tmph = tf.stack(h_list, 0)
        tmph_before, tmph_last = tf.split(tmph, [step-1, 1], 2)
        tmph_last = tf.squeeze( tmph_last, [2] )
        
        # ? bias nonlinear activation ?
        temp_logit = tf.reduce_sum( tmph_before*w_temp, 3 )
        temp_weight = tf.nn.softmax( temp_logit )
        
        tmph_cxt = tf.reduce_sum(tmph_before*tf.expand_dims(temp_weight, -1), 2)
        h_temp = tf.concat([tmph_cxt, tmph_last], 2)
        
        w_var = tf.get_variable('w_var', [h_dim*2, 1], initializer=tf.contrib.layers.xavier_initializer())
        b_var = tf.Variable( tf.random_normal([1]) )
        
        # ? bias nonlinear activation ?
        var_weight = tf.sigmoid( tf.tensordot(h_temp, w_var, axes=1) )
        
        h_var_list = tf.split(h_temp*var_weight, num_or_size_splits = len(h_list), axis = 0) 
        
    return tf.squeeze(tf.concat(h_var_list, 2)), tf.nn.l2_loss(w_var) + tf.nn.l2_loss(w_temp), [temp_weight, var_weight]


# unified temporal weighted attention 
def mv_attention_temp_concat_weight_decay( h_list, h_dim, scope, step, step_idx, decay_activation, att_type ):
    
    with tf.variable_scope(scope):
        
        tmph = tf.stack(h_list, 0)
        # [V B T-1 D], [V, B, 1, D]
        tmph_before, tmph_last = tf.split(tmph, [step-1, 1], 2)
        
        # -- temporal logits
        if att_type == 'loc':
            
            w_temp = tf.get_variable('w_temp', [len(h_list), 1, 1, h_dim], initializer=tf.contrib.layers.xavier_initializer())
            b_temp = tf.Variable( tf.random_normal([len(h_list), 1, 1, 1]) )
            
            # ? bias nonlinear activation ?
            #[V, B, T-1]
            temp_logit = tf.reduce_sum( tmph_before * w_temp, 3 ) 
            
        elif att_type == 'general':
            
            w_temp = tf.get_variable('w_temp', [len(h_list), 1, h_dim, h_dim],\
                                     initializer=tf.contrib.layers.xavier_initializer())
            b_temp = tf.Variable( tf.random_normal([len(h_list), 1, 1, 1]) )
            
            #[V, B, 1, D]
            tmp = tf.reduce_sum( tmph_last * w_temp, 3 )
            tmp = tf.expand_dims(tmp, 2)
        
            # ? bias nonlinear activation ?
            temp_logit = tf.reduce_sum(tmph_before * tmp, 3)
            
        elif att_type == 'concat':
            
            w_temp = tf.get_variable('w_temp', [len(h_list), 1, 1, h_dim*2],\
                                     initializer=tf.contrib.layers.xavier_initializer())
            b_temp = tf.Variable( tf.random_normal([len(h_list), 1, 1, 1]) )
            
            # concatenate tmph_before and tmph_last
            last_tile = tf.tile(tmph_last, [1, 1, step-1, 1])
            tmph_tile = tf.concat( [tmph_before, last_tile], 3 )
            
            # ? bias nonlinear activation ?
            temp_logit = tf.reduce_sum( tmph_tile * w_temp, 3 ) 
        
        else:
            print '[ERROR] attention type'
        
        
        # -- temporal decay weight
        w_decay = tf.get_variable('w_decay', [len(h_list), 1], initializer=tf.contrib.layers.xavier_initializer())
        w_decay = tf.square(w_decay)
        
        b_decay = tf.Variable( tf.random_normal([len(h_list), 1]) )
        step_idx = tf.reshape(step_idx, [1, step-1])
        
        #  new added
        # [V, T-1]
        v_step = tf.tile(step_idx, [len(h_list), 1])
        
        # [V, T-1]
        cutoff_decay = tf.Variable( tf.random_normal([len(h_list), 1]) )
        cutoff_decay = tf.sigmoid(cutoff_decay)*(step-1)
        
        # ? bias ?
        if decay_activation == 'exp':
            #temp_decay = tf.exp( tf.matmul(w_decay, -1*step_idx) )
            #temp_decay = tf.expand_dims(temp_decay, 1)
            
            # ? bias ?
            temp_decay = tf.exp(w_decay*(cutoff_decay - v_step))
            temp_decay = tf.expand_dims(temp_decay, 1)
            # [V, 1, T-1]
            
        elif decay_activation == 'sigmoid':
            #temp_decay = tf.sigmoid( tf.matmul(w_decay, -1*step_idx) )
            #temp_decay = tf.expand_dims(temp_decay, 1)
            
            # ? bias ?
            temp_decay = tf.sigmoid(w_decay*(cutoff_decay - v_step)) 
            temp_decay = tf.expand_dims(temp_decay, 1)
            # [V, 1, T-1]
            
        else:
            # no attention decay
            temp_weight = tf.nn.softmax( temp_logit )
            
            # temp_before [V B T-1 D], temp_weight [V B T-1]
            tmph_cxt = tf.reduce_sum(tmph_before*tf.expand_dims(temp_weight, -1), 2)
            tmph_last = tf.squeeze( tmph_last, [2] ) 
            
            h_temp = tf.concat([tmph_cxt, tmph_last], 2)
            h_var_list = tf.split(h_temp, num_or_size_splits = len(h_list), axis = 0) 
        
            return tf.squeeze(tf.concat(h_var_list, 2)), tf.nn.l2_loss(w_temp), temp_weight
        
        # -- decay on temporal logit
        # [V, B, T-1] * [V, 1, T-1]
        temp_logit  = temp_logit*temp_decay
        temp_weight = tf.nn.softmax( temp_logit )
        
        # -- attention weighted context
        # tmph_before [V B T-1 D]
        tmph_cxt = tf.reduce_sum( tmph_before*tf.expand_dims(temp_weight, -1), 2 )
        
        # [context, last hidden]
        # ?
        tmph_last = tf.squeeze( tmph_last, [2] )
        h_temp = tf.concat([tmph_last, tmph_cxt], 2)
        #h_temp = tmph_last
        
        h_var_list = tf.split(h_temp, num_or_size_splits = len(h_list), axis = 0) 
        
    return tf.squeeze(tf.concat(h_var_list, 2)), [tf.nn.l2_loss(w_temp),  tf.nn.l2_loss(w_decay)], temp_weight
    
# ---- multi-variate RNN ----

class tsLSTM_mv():
    
    def __init__(self, n_dense_dim_layers, n_lstm_dim_layers, n_steps, n_data_dim, session,\
                 lr, max_norm , n_batch_size, bool_residual, bool_att, decay, attention, \
                 l2_dense, l2_att):
        
        self.LEARNING_RATE = lr
        
        self.n_lstm_dim_layers = n_lstm_dim_layers
        
        self.N_STEPS    = n_steps
        self.N_DATA_DIM = n_data_dim
        
        self.n_dense_dim_layers = n_dense_dim_layers
        self.n_batch_size       = n_batch_size
        
        # placeholders
        self.x = tf.placeholder(tf.float32, [None, self.N_STEPS, self.N_DATA_DIM])
        self.y = tf.placeholder(tf.float32, [None, 1])
        self.keep_prob = tf.placeholder(tf.float32)
        
        steps = tf.constant( range(self.N_STEPS-2, -1, -1), dtype=tf.float32 )
        
        # begin to build the graph
        self.sess = session
        
        # residual connections
        if bool_residual == True:
            
            print '!! Wrongly use residual connection'
            
        # no residual connection 
        else:
            
            # no attention
            if bool_att == '':
                
                print ' --- Using no attention: '
                
                with tf.variable_scope('lstm'):
                    lstm_cell = MvLSTMCell( n_lstm_dim_layers[0], n_var = n_data_dim, \
                                           initializer=tf.contrib.layers.xavier_initializer() )
                    h, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = self.x, dtype = tf.float32)
            
                # obtain the last hidden state    
                tmp_hiddens = tf.transpose( h, [1,0,2] )
                h = tmp_hiddens[-1]
                
                # test
                self.test = tf.shape(h)
                
                h, regu_dense = plain_dense( h, n_lstm_dim_layers[-1], n_dense_dim_layers, 'dense', self.keep_prob )
                regu_all = l2_dense*regu_dense
            
            
            elif bool_att == 'temp':
                
                print ' --- Using temporal attention: '

                with tf.variable_scope('lstm'):
                    
                    lstm_cell = MvLSTMCell( n_lstm_dim_layers[0], n_var = n_data_dim ,\
                                           initializer = tf.contrib.layers.xavier_initializer() )
                    h, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = self.x, dtype = tf.float32)
                
                h_list = tf.split(h, [int(n_lstm_dim_layers[0]/self.N_DATA_DIM)]*self.N_DATA_DIM, 2)
                
                # test
                #self.test = tf.shape(h_list)
                
                h_att, regu_att, self.att = \
                mv_attention_temp_concat_weight_decay( h_list, int(n_lstm_dim_layers[0]/self.N_DATA_DIM),\
                                                       'att', self.N_STEPS, steps, decay, attention )
                # test
                self.test = tf.shape(h_att)
                
                # ?
                h, regu_dense = plain_dense( h_att, n_lstm_dim_layers[-1]*2, n_dense_dim_layers, 'dense', self.keep_prob )
                
                # ?
                if decay == '':
                    regu_all = l2_dense*regu_dense + l2_att*regu_att
                else:
                    regu_all = l2_dense*regu_dense + l2_att*(regu_att[0] + regu_att[1]) 
                
            elif bool_att == 'both':
                
                print ' --- Using both temporal and variate attention: '
                
                with tf.variable_scope('lstm'):
                    lstm_cell = MvLSTMCell( n_lstm_dim_layers[0], n_var = n_data_dim ,\
                                           initializer = tf.contrib.layers.xavier_initializer() )
                    h, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = self.x, dtype = tf.float32)
                
                h_list = tf.split(h, [int(n_lstm_dim_layers[0]/self.N_DATA_DIM)]*self.N_DATA_DIM, 2)
                 
                # test
                self.test = tf.shape(h_list)
                
                # ?
                h_att, regu_att, self.att = mv_attention_variate_temp_logit(h_list,int(n_lstm_dim_layers[0]/self.N_DATA_DIM),\
                                                                           'att', self.N_STEPS)
                
                h, regu_dense = plain_dense( h_att, n_lstm_dim_layers[-1]*2, n_dense_dim_layers, 'dense', self.keep_prob )
                
                # ?
                regu_all = l2_dense*regu_dense 
                #+ regu_att
                
            else:
                print '[ERROR] add attention'
        
        #dropout
        #last_hidden = tf.nn.dropout(last_hidden, self.keep_prob)
        with tf.variable_scope("output"):
            
            w = tf.get_variable('w', shape=[n_dense_dim_layers[-1], 1],\
                                     initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.zeros([1]))
            self.py = tf.matmul(h, w) + b
            
            self.regularization = regu_all + l2_dense*tf.nn.l2_loss(w)
        
        # add regularization in LSTM 
        self.regul_lstm = sum( tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() \
                                   if ("lstm" in tf_var.name and "input" in tf_var.name))
        
        # ?
        #self.regularization += self.regul_lstm

        
    def train_ini(self):  
        # loss function 
        self.cost = tf.reduce_mean( tf.square(self.y - self.py) ) + self.regularization
        
        self.optimizer = \
        tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)  
        
        self.init = tf.global_variables_initializer()
        self.sess.run( [self.init] )
        
    def train_batch(self, x_batch, y_batch, keep_prob ):
        _, c = self.sess.run([self.optimizer, self.cost],\
                              feed_dict={self.x:x_batch, self.y:y_batch, self.keep_prob:keep_prob })
        return c

#   initialize inference         
    def inference_ini(self):
#       denormalzied RMSE  
        self.rmse = tf.sqrt( tf.reduce_mean( tf.square( self.y - self.py ) ) )
        self.y_hat= self.py
        
#   infer givn testing data    
    def inference(self, x_test, y_test, keep_prob):
        return self.sess.run([self.rmse], feed_dict = {self.x:x_test, self.y:y_test,\
                                                                            self.keep_prob:keep_prob})
    
    def predict(self, x_test, y_test, keep_prob):
        return self.sess.run([self.y_hat], feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    
    def test_regularization(self, x_test, y_test, keep_prob):
        return self.sess.run([self.regularization], feed_dict = {self.x:x_test, self.y:y_test,\
                                                                            self.keep_prob:keep_prob})
    
    def test_attention(self, x_test, y_test, keep_prob):
        return self.sess.run([self.att],  feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    def testfunc(self, x_batch, y_batch, keep_prob ):
        tmpname= []
        for tf_var in tf.trainable_variables():
            tmpname.append( tf_var.name )
            
        #self.test, regul_lstm    
        return tmpname, self.sess.run([self.test],\
                            feed_dict={self.x:x_batch, self.y:y_batch, self.keep_prob:keep_prob })
  

'''
# --- backup code

def mv_attention_temp_logit_concat_exp_decay( h_list, h_dim, scope, step, step_idx ):
    
    with tf.variable_scope(scope):
        
        w_temp = tf.get_variable('w_temp', [len(h_list), 1, 1, h_dim], initializer=tf.contrib.layers.xavier_initializer())
        b_temp = tf.Variable( tf.random_normal([len(h_list), 1, 1, 1]) )
        
        tmph = tf.stack(h_list, 0)
        tmph_before, tmph_last = tf.split(tmph, [step-1, 1], 2)
        tmph_last = tf.squeeze( tmph_last, [2] )
        
        # ? bias nonlinear activation ?
        temp_logit = tf.reduce_sum( tmph_before*w_temp, 3 )
        
        # temporal decay
        temp_decay = tf.exp( -1.0*step_idx )
        temp_decay = tf.expand_dims(temp_decay, 0)
        temp_decay = tf.expand_dims(temp_decay, 0)
        
        temp_logit = temp_logit*temp_decay
        temp_weight = tf.nn.softmax( temp_logit )
        
        tmph_cxt = tf.reduce_sum( tmph_before*tf.expand_dims(temp_weight, -1), 2)
        h_temp = tf.concat([tmph_cxt, tmph_last], 2)
        
        h_var_list = tf.split(h_temp, num_or_size_splits = len(h_list), axis = 0) 
        
    return tf.squeeze(tf.concat(h_var_list, 2)), tf.nn.l2_loss(w_temp), temp_weight

def mv_attention_temp_location_weight_decay( h_list, h_dim, scope, step, step_idx, activation ):
    
    with tf.variable_scope(scope):
        
        w_temp = tf.get_variable('w_temp', [len(h_list), 1, 1, h_dim], initializer=tf.contrib.layers.xavier_initializer())
        b_temp = tf.Variable( tf.random_normal([len(h_list), 1, 1, 1]) )
        
        w_decay = tf.get_variable('w_decay', [len(h_list), 1], initializer=tf.contrib.layers.xavier_initializer())
        w_decay = tf.square(w_decay)
        
        b_decay = tf.Variable( tf.random_normal([len(h_list), 1]) )
        
        tmph = tf.stack(h_list, 0)
        tmph_before, tmph_last = tf.split(tmph, [step-1, 1], 2)
        
        # temporal logits
        # ? bias nonlinear activation ?
        temp_logit = tf.reduce_sum(tmph_before*w_temp, 3) + b_temp
        
        # temporal decay weight
        step_idx = tf.reshape(step_idx, [1, step-1])
        
        # ? bias ?
        if activation == 'exp':
            temp_decay = tf.exp( tf.matmul(w_decay, -1*step_idx) )
        elif activation == 'sigmoid':
            temp_decay = tf.sigmoid( tf.matmul(w_decay, -1*step_idx) )
        
        temp_decay = tf.expand_dims(temp_decay, 1)
        
        # decay on temporal logit
        temp_logit  = temp_logit*temp_decay
        temp_weight = tf.nn.softmax( temp_logit )
        
        # attention weighted context
        tmph_cxt = tf.reduce_sum( tmph_before*tf.expand_dims(temp_weight, -1), 2 )
        
        # [context, last hidden]
        # ?
        #h_temp = tmph_last
        tmph_last = tf.squeeze( tmph_last, [2] )
        h_temp = tf.concat([tmph_last, tmph_cxt], 2)
        
        h_var_list = tf.split(h_temp, num_or_size_splits = len(h_list), axis = 0) 
        
        # ?
        att_regul = tf.nn.l2_loss(w_temp) + tf.nn.l2_loss(w_decay)
        
    return tf.squeeze(tf.concat(h_var_list, 2)), att_regul, temp_weight

def mv_attention_temp_general_weight_decay( h_list, h_dim, scope, step, step_idx, activation ):
    
    with tf.variable_scope(scope):
        
        w_temp = tf.get_variable('w_temp', [len(h_list), 1, h_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
        b_temp = tf.Variable( tf.random_normal([len(h_list), 1, 1, 1]) )
        
        w_decay = tf.get_variable('w_decay', [len(h_list), 1], initializer=tf.contrib.layers.xavier_initializer())
        w_decay = tf.square(w_decay)
        b_decay = tf.Variable( tf.random_normal([len(h_list), 1]) )
        
        tmph = tf.stack(h_list, 0)
        tmph_before, tmph_last = tf.split(tmph, [step-1, 1], 2)
        
        # temporal logits
        #[V, B, 1, D]
        tmp = tf.reduce_sum(tmph_last * w_temp, 3)
        tmp = tf.expand_dims(tmp, 2)
        
        # ? bias nonlinear activation ?
        temp_logit = tf.nn.relu( tf.reduce_sum(tmph_before * tmp, 3) )
        
        # temporal decay weight
        step_idx = tf.reshape(step_idx, [1, step-1])
        
        # ? bias ?
        if activation == 'exp':
            temp_decay = tf.exp( tf.matmul(w_decay, -1*step_idx) )
        elif activation == 'sigmoid':
            temp_decay = tf.sigmoid( tf.matmul(w_decay, -1*step_idx) )
        
        temp_decay = tf.expand_dims(temp_decay, 1)
        
        # decay on temporal logits
        temp_logit  = temp_logit*temp_decay
        temp_weight = tf.nn.softmax( temp_logit )
        
        # [context, last hidden]
        tmph_last = tf.squeeze( tmph_last, [2] )
        tmph_cxt = tf.reduce_sum( tmph_before*tf.expand_dims(temp_weight, -1), 2 )
        
        # ?
        #h_temp = tmph_last
        h_temp = tf.concat([tmph_last, tmph_cxt], 2)
        
        h_var_list = tf.split(h_temp, num_or_size_splits = len(h_list), axis = 0) 
        
        # ?
        att_regul = tf.nn.l2_loss(w_temp) + tf.nn.l2_loss(w_decay)
        
    return tf.squeeze(tf.concat(h_var_list, 2)), att_regul, temp_weight

def mv_attention_temp_concat_weight_decay( h_list, h_dim, scope, step, step_idx, activation ):
    
    with tf.variable_scope(scope):
        
        w_temp = tf.get_variable('w_temp', [len(h_list), 1, 1, h_dim*2], initializer=tf.contrib.layers.xavier_initializer())
        b_temp = tf.Variable( tf.random_normal([len(h_list), 1, 1, 1]) )
        
        w_decay = tf.get_variable('w_decay', [len(h_list), 1], initializer=tf.contrib.layers.xavier_initializer())
        w_decay = tf.square(w_decay)
        
        b_decay = tf.Variable( tf.random_normal([len(h_list), 1]) )
        
        tmph = tf.stack(h_list, 0)
        tmph_before, tmph_last = tf.split(tmph, [step-1, 1], 2)
        
        # temporal logits
        
        # concatenate tmph_before and tmph_last
        last_tile = tf.tile(tmph_last, [1, 1, step-1, 1])
        tmph_before = tf.concat( [tmph_before, last_tile], 3 )
        # ? bias nonlinear activation ?
        temp_logit = tf.reduce_sum(tmph_before*w_temp, 3) + b_temp
        
        # temporal decay weight
        step_idx = tf.reshape(step_idx, [1, step-1])
        
        # ? bias ?
        if activation == 'exp':
            temp_decay = tf.exp( tf.matmul(w_decay, -1*step_idx) )
        elif activation == 'sigmoid':
            temp_decay = tf.sigmoid( tf.matmul(w_decay, -1*step_idx) )
        
        temp_decay = tf.expand_dims(temp_decay, 1)
        
        # decay on temporal logit
        temp_logit  = temp_logit*temp_decay
        temp_weight = tf.nn.softmax( temp_logit )
        
        # attention weighted context
        tmph_cxt = tf.reduce_sum( tmph_before*tf.expand_dims(temp_weight, -1), 2 )
        
        # [context, last hidden]
        # ?
        tmph_last = tf.squeeze( tmph_last, [2] )
        h_temp = tf.concat([tmph_last, tmph_cxt], 2)
        #h_temp = tmph_last
        
        h_var_list = tf.split(h_temp, num_or_size_splits = len(h_list), axis = 0) 
        
        # ?
        att_regul = tf.nn.l2_loss(w_temp) + tf.nn.l2_loss(w_decay)
        
    return tf.squeeze(tf.concat(h_var_list, 2)), att_regul, temp_weight
'''

'''
# residual connections

            # no attention
            if bool_att == '':
                
                print 'Using no attention: '
                
                with tf.variable_scope('lstm'):
                    lstm_cell = MvLSTMCell( n_lstm_dim_layers[0], n_var = 3 ,\
                                           initializer=tf.contrib.layers.xavier_initializer() )
                    # , initializer=tf.contrib.layers.xavier_initializer()
                    #, initializer= tf.contrib.keras.initializers.glorot_normal()
                    h, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = self.x, dtype = tf.float32)
            
                # obtain the last hidden state    
                tmp_hiddens = tf.transpose( h, [1,0,2] )
                h = tmp_hiddens[-1]
                
                # test
                self.test = tf.shape(h)
                
                h, regul = res_dense( h, n_lstm_dim_layers[-1], n_dense_dim_layers[0], len(n_dense_dim_layers),\
                                     'dense', self.keep_prob )
                regu_all = regul
            
            elif bool_att == 'temp':
                
                print 'Using temporal attention: '

                with tf.variable_scope('lstm'):
                    lstm_cell = MvLSTMCell( n_lstm_dim_layers[0], n_var = 3 ,\
                                           initializer=tf.contrib.layers.xavier_initializer() )
                    h, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = self.x, dtype = tf.float32)
                
                h_list = tf.split(h, num_or_size_splits = self.N_DATA_DIM, axis=2)
                
                #?
                self.test = tf.shape(h_list)
                #?
                
                #?
                h, regul = res_dense( h_att, n_lstm_dim_layers[-1]*2, n_dense_dim_layers[0], len(n_dense_dim_layers),\
                                     'dense', self.keep_prob )
                #?
                regu_all = regul
                
            elif bool_att == 'both':
                
                print 'Using both temporal and variate attention: '
                
                #, initializer=tf.contrib.layers.xavier_initializer()
                with tf.variable_scope('lstm'):
                    lstm_cell = MvLSTMCell( n_lstm_dim_layers[0] )
                    #, initializer= tf.contrib.keras.initializers.glorot_normal()
                    h, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = self.x, dtype = tf.float32)
                
                h_list = tf.split(h, num_or_size_splits = self.N_DATA_DIM, axis=2)
                 
                # test
                #self.test = tf.shape(h_list)
                
                h_att, regu_att, self.att = mv_attention_variate_temp_logit_speed_up(h_list,\
                                                                                  int(n_lstm_dim_layers[0]/self.N_DATA_DIM),\
                                                                                  'att', self.N_STEPS)
                
                h, regul = res_dense( h_att, n_lstm_dim_layers[-1]*2, n_dense_dim_layers[0], len(n_dense_dim_layers),\
                                     'dense', self.keep_prob )
                regu_all = regu_att + regul
            '''
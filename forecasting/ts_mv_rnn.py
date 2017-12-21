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
        #x = tf.nn.dropout(x, dropout_keep_prob)
        
        with tf.variable_scope(scope):
                # initilization
                w = tf.get_variable('w', [x_dim, hidden_dim], dtype = tf.float32,
                                          initializer = tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros([hidden_dim]))
                h = tf.nn.relu(tf.matmul(x, w) + b )

                regularization = tf.nn.l2_loss(w)
        #dropout
        h = tf.nn.dropout(h, dropout_keep_prob)
        
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

                regularization = tf.nn.l2_loss(w)
                
        #dropout
        #h = tf.nn.dropout(h, dropout_keep_prob)
        
        for i in range(1, len(dim_layers)):
            
            with tf.variable_scope(scope+str(i)):
                w = tf.get_variable('w', [dim_layers[i-1], dim_layers[i]],initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros( dim_layers[i] ))
                h = tf.nn.relu( tf.matmul(h, w) + b )
                
                regularization += tf.nn.l2_loss(w)
        
        return h, regularization

    
#---- Attention plain ----

# ref: a structured self attentive sentence embedding  
def attention_temporal_mlp( h, h_dim, att_dim, scope ):
    # tf.tensordot
    with tf.variable_scope(scope):
        
        w = tf.get_variable('w', [h_dim, att_dim], initializer=tf.contrib.layers.xavier_initializer())
        #? add bias?
        tmp_h = tf.nn.relu( tf.tensordot(h, w, axes=1) )

        w_logit = tf.get_variable('w_log', [att_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
        logit = tf.tensordot(tmp_h, w_logit, axes=1)
        
        alphas = tf.nn.softmax( tf.squeeze(logit) )
        
    return tf.reduce_sum(h*tf.expand_dims(alphas, -1), 1), alphas

def attention_temporal_logit( h, h_dim, scope ):
    # tf.tensordot
    with tf.variable_scope(scope):
        
        w = tf.get_variable('w', [h_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
        
        #? add bias 
        b = tf.Variable(tf.zeros([1, 1]))
        #? add nonlinear activiation 
        logit = tf.squeeze(tf.tensordot(h, w, axes=1))
        
        alphas = tf.nn.softmax( tf.squeeze(logit) )
        
    return tf.reduce_sum(h*tf.expand_dims(alphas, -1), 1), alphas, tf.nn.l2_loss(w) 

def attention_temporal_logit_context( h, h_dim, scope, step ):
    # tf.tensordot
    
    h_context, h_last = tf.split(h, [step-1, 1], 1)
    h_last = tf.squeeze(h_last)
    
    with tf.variable_scope(scope):
        
        w = tf.get_variable('w', [h_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
        
        #? add bias 
        b = tf.Variable(tf.zeros([1, 1]))
        #? add nonlinear activiation 
        logit = tf.squeeze(tf.tensordot(h_context, w, axes=1))
        
        alphas = tf.nn.softmax( tf.squeeze(logit) )
        
        context = tf.reduce_sum(h_context*tf.expand_dims(alphas, -1), 1)
        
    return tf.concat([context, h_last], 1), alphas, tf.nn.l2_loss(w) 
    
    
#---- plain RNN ----

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
                h, self.att, regu = attention_temporal_logit( h, n_lstm_dim_layers[0], 'att' )
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
                
                # ?
                #h, self.att, regu = attention_temporal_logit( h, n_lstm_dim_layers[-1], 'att' )
                h, self.att, regu = attention_temporal_logit_context( h, n_lstm_dim_layers[-1], 'att', self.N_STEPS )
                
                h, regul = plain_dense( h, n_lstm_dim_layers[-1]*2, n_dense_dim_layers, 'dense', self.keep_prob )
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
    
    
    
#---- Attention Sep ----
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
def attention_variate_temp_mlp_speed_up( h_list, h_dim, scope, step ):
    
    with tf.variable_scope(scope):
        
        w_temp = tf.get_variable('w_temp', [len(h_list), 1, 1, h_dim], initializer=tf.contrib.layers.xavier_initializer())
        b_temp = tf.Variable( tf.random_normal([len(h_list), 1, 1, 1]) )
        
        tmph = tf.stack(h_list, 0)
        tmph_before, tmph_last = tf.split(tmph, [step-1, 1], 2)
        tmph_last = tf.squeeze( tmph_last, [2] )
        
        # ? bias nonlinear activation?
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


# attention_temporal_logit( h, h_dim, step, scope ):
# [#variate, batch_size, steps, dim]
#assume each time series has the hidden of the same dimension
def attention_variate_logit( h_list, h_dim, scope, step ):
    
    hh = []
    att = []
     
    with tf.variable_scope(scope):
        w_var = tf.get_variable('w_', [h_dim*2, 1], initializer=tf.contrib.layers.xavier_initializer())
        b_var = tf.Variable( tf.random_normal([1]) )

        for i in range(len(h_list)):
            temp_att_feature, temp_att, tmp_regu = attention_temporal_logit_context(h_list[i], h_dim, scope + str(i), step)
        # [B, 2*D]
        
            if i==0:
                regul = tmp_regu
            else:
                regul += tmp_regu
            
            # ? bias nonlinear activation?
            var_weight = tf.sigmoid( tf.matmul(temp_att_feature, w_var) + b_var)
        # [B, 1]
            var_feature = temp_att_feature*var_weight
        
            hh.append(var_feature)
            att.append(temp_att)
            
    #shape of v: [ batch_size, variate*h_dim]
        v = tf.concat(hh, 1)
    
    return v, tf.nn.l2_loss(w_var) + regul, att

def attention_temp_logit_speed_up( h_list, h_dim, scope, step ):
    
    with tf.variable_scope(scope):
        
        w_temp = tf.get_variable('w_', [len(h_list), 1, 1, h_dim], initializer=tf.contrib.layers.xavier_initializer())
        b_temp = tf.Variable( tf.random_normal([len(h_list), 1, 1, 1]) )
        
        tmph = tf.stack(h_list, 0)
        tmph_before, tmph_last = tf.split(tmph, [step-1, 1], 2)
        tmph_last = tf.squeeze( tmph_last, [2] )
        
        # ? bias nonlinear activation?
        temp_logit = tf.reduce_sum( tmph_before*w_temp, 3 )
        temp_weight = tf.nn.softmax( temp_logit )
        
        tmph_cxt = tf.reduce_sum( tmph_before*tf.expand_dims(temp_weight, -1), 2)
        
        #?
        h_temp = tf.concat([tmph_cxt, tmph_last], 2)
        #h_temp = tmph_cxt + tmph_last
        
        h_var_list = tf.split(h_temp, num_or_size_splits = len(h_list), axis = 0) 
        
    return tf.squeeze(tf.concat(h_var_list, 2)), tf.nn.l2_loss(w_temp), temp_weight

def attention_variate_temp_logit_speed_up( h_list, h_dim, scope, step ):
    
    with tf.variable_scope(scope):
        
        w_temp = tf.get_variable('w_temp', [len(h_list), 1, 1, h_dim], initializer=tf.contrib.layers.xavier_initializer())
        b_temp = tf.Variable( tf.random_normal([len(h_list), 1, 1, 1]) )
        
        tmph = tf.stack(h_list, 0)
        tmph_before, tmph_last = tf.split(tmph, [step-1, 1], 2)
        tmph_last = tf.squeeze( tmph_last, [2] )
        
        # ? bias nonlinear activation?
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
        
        if bool_att == 'both':
            
            #h, regu, self.att = attention_variate_logit( concat_h, n_lstm_dim_layers[-1], 'attention', self.N_STEPS )
            
            h, regu, self.att = attention_variate_temp_logit_speed_up( concat_h, n_lstm_dim_layers[-1], 'attention',\
                                                                      self.N_STEPS )
            # dense layers 
            h, regul = plain_dense(h, n_lstm_dim_layers[-1]*self.N_DATA_DIM*2, n_dense_dim_layers, 'dense', self.keep_prob)
            self.regularization = regul + regu
        
        elif bool_att == 'temp':
            
            h, regu, self.att = attention_temp_logit_speed_up( concat_h, n_lstm_dim_layers[-1], 'attention', self.N_STEPS )
            
            # dense layers 
            h, regul = plain_dense(h, n_lstm_dim_layers[-1]*self.N_DATA_DIM*2, n_dense_dim_layers, 'dense', self.keep_prob)
            self.regularization = regul + regu
            
        else: 
            # no attention
            
            # hidden space merge
            h = tf.concat(concat_h, 1)
            h, regul = plain_dense(h, n_lstm_dim_layers[-1]*self.N_DATA_DIM, n_dense_dim_layers, 'dense', self.keep_prob)
            self.regularization = regul
        
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
        
        # ? bias nonlinear activation?
        temp_logit = tf.reduce_sum( tmph*w_temp+b_temp, 3 )
        temp_weight = tf.nn.softmax( temp_logit )
        
        #tmph_cxt = tf.reduce_sum( tmph_before*tf.expand_dims(temp_weight, -1), 2)
        #tf.concat([tmph_cxt, tmph_last], 2)
        h_temp = tf.reduce_sum( tmph*tf.expand_dims(temp_weight, -1), 2)
        
        h_var_list = tf.split(h_temp, num_or_size_splits = len(h_list), axis = 0) 
        
    return tf.squeeze(tf.concat(h_var_list, 2)), tf.nn.l2_loss(w_temp), temp_weight
#tf.shape(tf.squeeze(tf.concat(h_var_list, 2)))

def mv_attention_temp_logit_add( h_list, h_dim, scope, step ):
    
    with tf.variable_scope(scope):
        
        w_temp = tf.get_variable('w_', [len(h_list), 1, 1, h_dim], initializer=tf.contrib.layers.xavier_initializer())
        b_temp = tf.Variable( tf.random_normal([len(h_list), 1, 1, 1]) )
        
        tmph = tf.stack(h_list, 0)
        tmph_before, tmph_last = tf.split(tmph, [step-1, 1], 2)
        tmph_last = tf.squeeze( tmph_last, [2] )
        
        # ? bias nonlinear activation?
        temp_logit = tf.reduce_sum( tmph_before*w_temp+b_temp, 3 )
        temp_weight = tf.nn.softmax( temp_logit )
        
        tmph_cxt = tf.reduce_sum( tmph_before*tf.expand_dims(temp_weight, -1), 2)
        
        h_temp = tmph_cxt + tmph_last
        
        h_var_list = tf.split(h_temp, num_or_size_splits = len(h_list), axis = 0) 
        
    return tf.squeeze(tf.concat(h_var_list, 2)), tf.nn.l2_loss(w_temp), temp_weight


#[hi, ht]
def mv_attention_temp_logit_concat( h_list, h_dim, scope, step ):
    
    with tf.variable_scope(scope):
        
        w_temp = tf.get_variable('w_', [len(h_list), 1, 1, h_dim], initializer=tf.contrib.layers.xavier_initializer())
        b_temp = tf.Variable( tf.random_normal([len(h_list), 1, 1, 1]) )
        
        tmph = tf.stack(h_list, 0)
        tmph_before, tmph_last = tf.split(tmph, [step-1, 1], 2)
        tmph_last = tf.squeeze( tmph_last, [2] )
        
        # ? bias nonlinear activation?
        temp_logit = tf.reduce_sum( tmph_before*w_temp + b_temp, 3 )
        temp_weight = tf.nn.softmax( temp_logit )
        
        tmph_cxt = tf.reduce_sum( tmph_before*tf.expand_dims(temp_weight, -1), 2)
        h_temp = tf.concat([tmph_cxt, tmph_last], 2)
        
        h_var_list = tf.split(h_temp, num_or_size_splits = len(h_list), axis = 0) 
        
    return tf.squeeze(tf.concat(h_var_list, 2)), tf.nn.l2_loss(w_temp), temp_weight
#tf.shape(tf.squeeze(tf.concat(h_var_list, 2)))


def mv_attention_temp_logit_concat_expdecay( h_list, h_dim, scope, step, step_idx ):
    
    with tf.variable_scope(scope):
        
        w_temp = tf.get_variable('w_', [len(h_list), 1, 1, h_dim], initializer=tf.contrib.layers.xavier_initializer())
        b_temp = tf.Variable( tf.random_normal([len(h_list), 1, 1, 1]) )
        
        tmph = tf.stack(h_list, 0)
        tmph_before, tmph_last = tf.split(tmph, [step-1, 1], 2)
        tmph_last = tf.squeeze( tmph_last, [2] )
        
        # ? bias nonlinear activation?
        temp_logit = tf.reduce_sum( tmph_before*w_temp+b_temp, 3 )
        
        temp_decay = tf.exp( -1*step_idx )
        temp_decay = tf.expand_dims(temp_decay, 0)
        temp_decay = tf.expand_dims(temp_decay, 0)
        
        temp_logit = temp_logit*temp_decay
        temp_weight = tf.nn.softmax( temp_logit )
        
        tmph_cxt = tf.reduce_sum( tmph_before*tf.expand_dims(temp_weight, -1), 2)
        h_temp = tf.concat([tmph_cxt, tmph_last], 2)
        
        h_var_list = tf.split(h_temp, num_or_size_splits = len(h_list), axis = 0) 
        
    return tf.squeeze(tf.concat(h_var_list, 2)), tf.nn.l2_loss(w_temp), temp_weight


'''
def mv_attention_variate_temp_logit( h_list, h_dim, scope, step ):
    
    with tf.variable_scope(scope):
        
        w_temp = tf.get_variable('w_temp', [len(h_list), 1, 1, h_dim], initializer=tf.contrib.layers.xavier_initializer())
        b_temp = tf.Variable( tf.random_normal([len(h_list), 1, 1, 1]) )
        
        tmph = tf.stack(h_list, 0)
        tmph_before, tmph_last = tf.split(tmph, [step-1, 1], 2)
        tmph_last = tf.squeeze( tmph_last, [2] )
        
        # ? bias nonlinear activation?
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
'''

# ---- multi-variate RNN ----

class tsLSTM_mv():
    
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
        
        steps = tf.constant( range(self.N_STEPS-1, -1, -1) )
        
        # begin to build the graph
        self.sess = session
        
        # residual connections
        if bool_residual == True:
            
            if bool_att == 'temp':

                with tf.variable_scope('lstm'):
                    lstm_cell = MvLSTMCell( n_lstm_dim_layers[0], n_var = 1 ,\
                                           initializer=tf.contrib.layers.xavier_initializer() )
                    #, initializer=tf.contrib.layers.xavier_initializer()
                    #, initializer= tf.contrib.keras.initializers.glorot_normal()
                    h, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = self.x, dtype = tf.float32)
                
                h_list = tf.split(h, [int(n_lstm_dim_layers[0]/self.N_DATA_DIM)]*self.N_DATA_DIM, 2)
                
                #?
                self.test = tf.shape(h_list)
                #?
                #h_att, regu_att, self.att = mv_attention_temp_logit_add(h_list,int(n_lstm_dim_layers[0]/self.N_DATA_DIM),\
                #                                                             'att', self.N_STEPS)
                
                #h_att, regu_att, self.att = mv_attention_temp_logit_concat(h_list,int(n_lstm_dim_layers[0]/self.N_DATA_DIM),\
                #                                                             'att', self.N_STEPS)
                
                #h_att, regu_att, self.att = mv_attention_temp_logit_all(h_list,int(n_lstm_dim_layers[0]/self.N_DATA_DIM),\
                #                                                             'att', self.N_STEPS)
                
                h_att, regu_att, self.att = mv_attention_temp_logit_concat_expdecay(h_list,\
                                                                                 int(n_lstm_dim_layers[0]/self.N_DATA_DIM),\
                                                                                 'att', self.N_STEPS, steps)
                
                #?
                h, regul = res_dense( h_att, n_lstm_dim_layers[-1]*2, n_dense_dim_layers[0], len(n_dense_dim_layers),\
                                     'dense', self.keep_prob )
                #?
                regu_all = regul
                
            elif bool_att == 'both':
                
                #, initializer=tf.contrib.layers.xavier_initializer()
                with tf.variable_scope('lstm'):
                    lstm_cell = MvLSTMCell( n_lstm_dim_layers[0] )
                    #, initializer= tf.contrib.keras.initializers.glorot_normal()
                    h, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = self.x, dtype = tf.float32)
                
                h_list = tf.split(h, [int(n_lstm_dim_layers[0]/self.N_DATA_DIM)]*self.N_DATA_DIM, 2)
                 
                # test
                #self.test = tf.shape(h_list)
                
                h_att, regu_att, self.att = mv_attention_variate_temp_logit_speed_up(h_list,\
                                                                                  int(n_lstm_dim_layers[0]/self.N_DATA_DIM),\
                                                                                  'att', self.N_STEPS)
                
                h, regul = res_dense( h_att, n_lstm_dim_layers[-1]*2, n_dense_dim_layers[0], len(n_dense_dim_layers),\
                                     'dense', self.keep_prob )
                regu_all = regu_att + regul
            
            else:
                # no attention
                with tf.variable_scope('lstm'):
                    lstm_cell = MvLSTMCell( n_lstm_dim_layers[0], n_var = 1 ,\
                                           initializer=tf.contrib.layers.xavier_initializer() )
                    # , initializer=tf.contrib.layers.xavier_initializer()
                    #, initializer= tf.contrib.keras.initializers.glorot_normal()
                    h, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = self.x, dtype = tf.float32)
            
                # obtain the last hidden state    
                tmp_hiddens = tf.transpose( h, [1,0,2]  )
                h = tmp_hiddens[-1]
                
                # test
                self.test = tf.shape(h)
                
                h, regul = res_dense( h, n_lstm_dim_layers[-1], n_dense_dim_layers[0], len(n_dense_dim_layers),\
                                     'dense', self.keep_prob )
                regu_all = regul
        
        
        # no residual connection 
        else:
            
            if bool_att == 'temp':

                with tf.variable_scope('lstm'):
                    lstm_cell = MvLSTMCell( n_lstm_dim_layers[0] )
                    #, initializer=tf.contrib.layers.xavier_initializer()
                    #, initializer= tf.contrib.keras.initializers.glorot_normal()
                    h, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = self.x, dtype = tf.float32)
                
                h_list = tf.split(h, [int(n_lstm_dim_layers[0]/self.N_DATA_DIM)]*self.N_DATA_DIM, 2)
                
                h_att, regu_att, self.att = mv_attention_temp_logit_speed_up(h_list,int(n_lstm_dim_layers[0]/self.N_DATA_DIM),\
                                                                             'att', self.N_STEPS)
                
                h, regul = plain_dense( h_att, n_lstm_dim_layers[-1]*2, n_dense_dim_layers, 'dense', self.keep_prob )
                regu_all = regu_att + regul
                
            elif bool_att == 'both':
                
                #, initializer=tf.contrib.layers.xavier_initializer()
                with tf.variable_scope('lstm'):
                    lstm_cell = MvLSTMCell( n_lstm_dim_layers[0] )
                    #, initializer= tf.contrib.keras.initializers.glorot_normal()
                    h, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = self.x, dtype = tf.float32)
                
                h_list = tf.split(h, [int(n_lstm_dim_layers[0]/self.N_DATA_DIM)]*self.N_DATA_DIM, 2)
                 
                # test
                #self.test = tf.shape(h_list)
                
                h_att, regu_att, self.att = mv_attention_variate_temp_logit_speed_up(h_list,\
                                                                                  int(n_lstm_dim_layers[0]/self.N_DATA_DIM),\
                                                                                  'att', self.N_STEPS)
                
                h, regul = plain_dense( h_att, n_lstm_dim_layers[-1]*2, n_dense_dim_layers, 'dense', self.keep_prob )
                regu_all = regu_att + regul
            
            else:
                # no attention
                with tf.variable_scope('lstm'):
                    lstm_cell = MvLSTMCell( n_lstm_dim_layers[0] )
                    # , initializer=tf.contrib.layers.xavier_initializer()
                    #, initializer= tf.contrib.keras.initializers.glorot_normal()
                    h, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = self.x, dtype = tf.float32)
            
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
        
        
        
        # add regularization in LSTM 
        self.regul_lstm = sum( tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() \
                                   if ("lstm" in tf_var.name and "input" in tf_var.name))
        # ?
        #self.regularization += self.regul_lstm

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
        return self.sess.run([self.rmse], feed_dict = {self.x:x_test, self.y:y_test,\
                                                                            self.keep_prob:keep_prob})
    
    def predict(self, x_test, y_test, keep_prob):
        return self.sess.run([self.y_hat], feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    
    def test_regularization(self, x_test, y_test, keep_prob):
        return self.sess.run([self.L2*self.regularization], feed_dict = {self.x:x_test, self.y:y_test,\
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
# ---- training process ----

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
    # always in formt [#, time_steps, data dimension]
    if len(tr_shape)==2:
        xtrain = np.expand_dims( xtrain, 2 )
        xtest  = np.expand_dims( xtest,  2 )
    elif len(tr_shape)==3:
        xtrain = np.reshape( xtrain, tr_shape )
        xtest  = np.reshape( xtest,  ts_shape )

    ytrain = np.expand_dims( ytrain, 1 ) 
    ytest  = np.expand_dims( ytest,  1 )

    print np.shape(xtrain), np.shape(ytrain), np.shape(xtest), np.shape(ytest)

    
# --- training log --- 
    log_file   = "res/ts_rnn_tf.txt"
    model_file = "res/model/ts_rnn_tf-{epoch:02d}.hdf5"

#   clean logs
    with open(log_file, "w") as text_file:
        text_file.close()

# --- network set-up ---
    para_input_dim = np.shape(xtrain)[-1]
    para_win_size =  np.shape(xtrain)[1]
    
    # if residual layers are used, keep all dimensions the same 
    para_lstm_dims   = [120, 120, 120]
    #para_lstm_dims   = [96, 96, 96]
    para_dense_dims  = [32, 32, 32]


    para_lr = 0.001
    para_batch_size = 128
    
    para_l2 = 0.008
    para_max_norm = 0.0
    para_keep_prob = 0.8

    para_is_stateful = False
    para_n_epoch = 500
    para_bool_residual = True

#--- build and train the model ---
    
    tf.reset_default_graph()
    
    with tf.Session() as sess:
        
        #reg = tsLSTM_discriminative(para_dense_dims, para_lstm_dims, \
        #                            para_win_size,   para_input_dim, sess, \
        #                            para_lr, para_l2,para_max_norm, para_batch_size, para_bool_residual)
        
                
        #reg = tsLSTM_seperate_mv(para_dense_dims, para_lstm_dims, \
        #                         para_win_size,   para_input_dim, sess, \
        #                         para_lr, para_l2,para_max_norm, para_batch_size)
        
        reg = tsLSTM_mv(para_dense_dims, para_lstm_dims, \
                                 para_win_size,   para_input_dim, sess, \
                                 para_lr, para_l2,para_max_norm, para_batch_size, para_bool_residual)
        
        # initialize the network
        reg.train_ini()
        reg.inference_ini()
    
        total_cnt   = np.shape(xtrain)[0]
        total_iter = int(total_cnt/para_batch_size)
        total_idx = range(total_cnt)
        
        # training epoches 
        for epoch in range(para_n_epoch):
            
            tmpc = 0.0
            np.random.shuffle(total_idx)
            
            for i in range(total_iter):
                
                # shuffle training data
                batch_idx = total_idx[ i*para_batch_size: (i+1)*para_batch_size ] 
                batch_x = xtrain[ batch_idx ]
                batch_y = ytrain[ batch_idx ]            
                
                tmpc += reg.train_batch( batch_x, batch_y, para_keep_prob)
        
            #if epoch%para_eval_byepoch != 0:
            #    continue
    
            tmp_test_acc  = reg.inference( xtest, ytest,  para_keep_prob) 
            tmp_train_acc = reg.inference( xtrain,ytrain, para_keep_prob)
            
            print "At epoch %d: loss %f, train %f, test %f\n" % ( epoch, tmpc*1.0/total_iter, \
                                                                  tmp_train_acc[0], tmp_test_acc[0] )  
            
            with open(log_file, "a") as text_file:
                    text_file.write("At epoch %d: loss %f, train %f, test %f\n" % ( epoch, tmpc*1.0/total_iter, \
                                                                                   tmp_train_acc[0], tmp_test_acc[0] ))  
        
        
        print "Optimization Finished!"
        
'''
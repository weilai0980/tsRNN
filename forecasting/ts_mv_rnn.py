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
                
                # ?
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
                
                # ?
                h = tf.nn.relu( tf.matmul(h, w) + b )
                
                #?
                regularization += tf.nn.l2_loss(w)
                #regularization += tf.reduce_sum(tf.abs(w))
                
        return h, regularization

def plain_dense_leaky(x, x_dim, dim_layers, scope, dropout_keep_prob, alpha):
    
        #dropout
        x = tf.nn.dropout(x, dropout_keep_prob)
        
        with tf.variable_scope(scope):
                # initilization
                w = tf.get_variable('w', [x_dim, dim_layers[0]], dtype=tf.float32,\
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros([dim_layers[0]]))
                
                # ?
                tmp_h = tf.matmul(x, w) + b 
                h = tf.maximum( alpha*tmp_h, tmp_h )

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
                
                # ?
                tmp_h = tf.matmul(h, w) + b 
                h = tf.maximum( alpha*tmp_h, tmp_h )
                
                #?
                regularization += tf.nn.l2_loss(w)
                #regularization += tf.reduce_sum(tf.abs(w))
                
        return h, regularization
    

    
# ---- Attention plain ----

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


def attention_temp_logit( h, h_dim, scope, step ):
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
        
        h, _ = plain_lstm( self.x, n_lstm_dim_layers, 'lstm', self.keep_prob )
            
            
        if bool_att == 'temp':
            
            print ' --- Plain RNN using temporal attention:  '
                
            h, self.att, regu = attention_temp_logit( h, n_lstm_dim_layers[-1], 'att', self.N_STEPS )
                
            h, regul = plain_dense( h, n_lstm_dim_layers[-1]*2, n_dense_dim_layers, 'dense', self.keep_prob )
                
            #?
            regu_all = regu + regul 
            
        else:
            
            print ' --- Plain RNN using no attention:  '
            
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
            # ?
            self.regularization = regu_all + tf.nn.l2_loss(w)
            
    def train_ini(self):
        
        # loss function 
        self.error_mean = tf.reduce_mean( tf.square(self.y - self.py) )
        self.error_sum  = tf.reduce_sum( tf.square(self.y - self.py) )
        
        # ?
        self.loss = self.error_mean + self.L2*self.regularization
        
        # optimizer 
        self.optimizer = \
        tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.loss)  
        
        # initilization 
        self.init = tf.global_variables_initializer()
        self.sess.run( [self.init] )
        
    def train_batch(self, x_batch, y_batch, keep_prob ):
        
        _, c, err_sum = self.sess.run([self.optimizer, self.loss, self.error_sum ],\
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
        return self.sess.run([self.py, self.rmse, self.mae, self.mape], \
                             feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    def predict(self, x_test, y_test, keep_prob):
        return self.sess.run([self.py], feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    def test_attention(self, x_test, y_test, keep_prob):
        return self.sess.run( self.att,  feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    
    
# ---- Attention for Sep RNN ----

def sep_attention_temp_logit( h_list, h_dim, scope, step ):
    
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
        # [V B 2D]
        h_temp = tf.concat([tmph_cxt, tmph_last], 2)
        #h_temp = tmph_cxt + tmph_last
        
        h_var_list = tf.split(h_temp, num_or_size_splits = len(h_list), axis = 0) 
        
    return tf.squeeze(tf.concat(h_var_list, 2)), tf.nn.l2_loss(w_temp), temp_weight

def sep_attention_variate_temp_logit( h_list, h_dim, scope, step ):
    
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

        
# ---- separate RNN ----

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
            
            h, att_regu, self.att = sep_attention_temp_logit( concat_h, n_lstm_dim_layers[-1], 'attention', self.N_STEPS )
            
            # dense layers 
            h, regul = plain_dense(h, n_lstm_dim_layers[-1]*self.N_DATA_DIM*2, n_dense_dim_layers, 'dense', self.keep_prob)
            self.regularization = regul + att_regu
            
        elif bool_att == 'both':
            
            h, att_regu, self.att = sep_attention_variate_temp_logit( concat_h, n_lstm_dim_layers[-1],\
                                                                     'attention',self.N_STEPS )
            
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
        self.loss = tf.reduce_mean( tf.square(self.y - self.py) ) + self.L2 * self.regularization
        
        self.optimizer = \
        tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.loss)  
        
        self.init = tf.global_variables_initializer()
        self.sess.run( [self.init] )
        
    def train_batch(self, x_batch, y_batch, keep_prob ):
        _, c = self.sess.run([self.optimizer, self.cost],feed_dict={self.x:x_batch, self.y:y_batch, self.keep_prob:keep_prob})
        return c

#   initialize inference         
    def inference_ini(self):

#       denormalzied RMSE  
        self.rmse = tf.sqrt( tf.reduce_mean(tf.square(self.y - self.py)) )
        
#   infer givn testing data 
    def inference(self, x_test, y_test, keep_prob):
        return self.sess.run([self.rmse], feed_dict={self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    def test_attention(self, x_test, y_test, keep_prob):
        return self.sess.run([self.att],  feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    def testfunc(self, x_batch, y_batch, keep_prob ):
        return self.sess.run([self.test],\
                              feed_dict={self.x:x_batch, self.y:y_batch, self.keep_prob:keep_prob })

    
# ---- Attention for MV-RNN ----

def mv_pooling_temp( h_var, pool_type, step ):
    
    # [V B T D]
    # [V B T-1 D], [V B 1 D]
    tmph_before, tmph_last = tf.split(h_var, [step-1, 1], 2)
    
    if pool_type == 'max':
        tmph_before_reduce = tf.reduce_max(tmph_before, 2)
    elif pool_type == 'average':
        tmph_before_reduce = tf.reduce_mean(tmph_before, 2)
    else:
        print '[ERROR] pooling type'
            
    #[V B D]
    tmph_last_reduce = tf.squeeze(tmph_last, 2)
    
    #[V B 2D]
    return tf.concat([tmph_before_reduce, tmph_last_reduce], -1)


# variate attention based on the temporal weighted hiddens
def mv_attention_variate( h_temp, var_dim, scope, num_vari, att_type ):
    
    # [V B D]
    with tf.variable_scope(scope):
        
        # softmax
        if att_type == 'softmax':
            
            # [V B D]
            w_var = tf.get_variable('w_var', [var_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
            b_var = tf.Variable( tf.random_normal([1]) )
            
            #[V-1 B D], [1 B D]
            h_indep, h_tar = tf.split(h_temp, num_or_size_splits = [num_vari - 1, 1], axis = 0)
            
            #? bias nonlinear activation ?
            # [V-1 B 1] = [V-1 B D]*[D 1]
            logits = tf.transpose( tf.tensordot(h_indep, w_var, axes=1) + b_var, [1, 0, 2] )
            # [B V-1 1]
            var_weight = tf.nn.softmax( logits , dim = 1 )
            
            # [B V-1 D]
            h_indep_trans = tf.transpose(h_indep, [1, 0, 2])
                        
            # sum-up, [B D]
            h_indep_weighted = tf.reduce_sum( h_indep_trans*var_weight, 1 )
            
            # [B D]
            h_tar_squeeze = tf.squeeze( h_tar, [0] )
            
            # [B 2D]
            h_res = tf.concat([h_indep_weighted, h_tar_squeeze], 1)
            
        elif att_type == 'softmax-all':
            
            # [V B D]
            w_var = tf.get_variable('w_var', [var_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
            b_var = tf.Variable( tf.random_normal([1]) )
            
            #? bias nonlinear activation ?
            # [V B 1] = [V B D]*[D 1]
            logits = tf.transpose( tf.tensordot(h_temp, w_var, axes=1) + b_var, [1, 0, 2] )
            # [B V 1]
            var_weight = tf.nn.softmax( logits , dim = 1 )
            
            # for place holder
            h_res = h_temp
        
        elif att_type == 'sigmoid':
            
            # [V B D]
            w_var = tf.get_variable('w_var', [var_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
            b_var = tf.Variable( tf.random_normal([1]) )
            
            #[V-1 B D], [1 B D]
            h_indep, h_tar = tf.split(h_temp, num_or_size_splits = [num_vari - 1, 1], axis = 0)
            
            # ? bias nonlinear activation ?
            # [V-1 B 1]
            var_weight = tf.sigmoid( tf.tensordot(h_indep, w_var, axes=1) + b_var )
            
            # [V-1 B D]
            h_indep_weighted = h_indep * var_weight
            
            # [V B D]
            h_weighted = tf.concat([h_indep_weighted, h_tar], 0)
            
            # [V B D]
            h_var_list = tf.split(h_weighted, num_or_size_splits = num_vari, axis = 0) 
            # [B 2H]
            h_res = tf.squeeze(tf.concat(h_var_list, 2)) 
            
            var_weight = tf.transpose(var_weight, [1,0,2])
            
        elif att_type == 'mlp':
            
            interm_dim =  var_dim/2
            
            # [D d]
            w_dense = tf.get_variable('w_dense', [ var_dim, interm_dim ],\
                                     initializer=tf.contrib.layers.xavier_initializer())
            b_dense = tf.Variable( tf.zeros([ interm_dim, ]) )
            
            
            # [V-1 B D], [1 B D]
            h_indep, h_tar = tf.split(h_temp, num_or_size_splits = [num_vari - 1, 1], axis = 0)
            
            # ? non-linear activation
            # [V-1 B d]
            h_interm_indep = tf.nn.relu( tf.tensordot(h_indep, w_dense, axes=1) + b_dense )
            # [1 B d]
            h_interm_tar = tf.nn.relu( tf.tensordot(h_tar, w_dense, axes=1) + b_dense )
            # [V-1 B d]
            h_interm = h_interm_indep + h_interm_tar
            
            
            w_var = tf.get_variable('w_var', [interm_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
            b_var = tf.Variable( tf.random_normal([1]) )
            
            # [V-1 B d] * [d 1]  [B V-1 1]
            logits = tf.transpose( tf.tensordot(h_interm, w_var, axes=1) + b_var, [1, 0, 2] )
            # [B V-1 1]
            var_weight = tf.nn.softmax( logits , dim = 1 )
            
            # [B V-1 D]
            h_indep_trans = tf.transpose(h_indep, [1, 0, 2])
            # sum-up, [B D]
            h_indep_weighted = tf.reduce_sum( h_indep_trans*var_weight, 1 )
            
            # [B D]
            h_tar_squeeze = tf.squeeze( h_tar, [0] )
            
            # [B 2D]
            h_res = tf.concat([h_indep_weighted, h_tar_squeeze], 1)
            
         
        else:
            print '[ERROR] variable attention type'
        
    return h_res, tf.nn.l2_loss(w_var), var_weight


# variate attention based on the temporal weighted hiddens
def mv_attention_variate_temp( h_temp, h_dim, scope, num_vari, att_type ):
    
    with tf.variable_scope(scope):
        
        # softmax
        if att_type == 'sum':
            
            # [V B 2D]
            w_var = tf.get_variable('w_var', [h_dim*2, 1], initializer=tf.contrib.layers.xavier_initializer())
            b_var = tf.Variable( tf.random_normal([1]) )
            
            #[V-1 B 2D], [1 B 2D]
            h_indep, h_tar = tf.split(h_temp, num_or_size_splits = [num_vari - 1, 1], axis = 0)
            
            #? bias nonlinear activation ?
            # [V-1 B 1] = [V-1 B 2D]*[2D 1]
            logits = tf.transpose( tf.tensordot(h_indep, w_var, axes=1) + b_var, [1, 0, 2] )
            # [B V-1 1]
            var_weight = tf.nn.softmax( logits , dim = 1 )
            
            # [B V-1 2D]
            h_indep_trans = tf.transpose(h_indep, [1, 0, 2])
                        
            # sum-up, [B 2D]
            h_indep_weighted = tf.reduce_sum( h_indep_trans*var_weight, 1 )
            
            # [B 2D]
            h_tar_squeeze = tf.squeeze( h_tar, [0] )
            
            # [B 4D]
            h_res = tf.concat([h_indep_weighted, h_tar_squeeze], 1)
        
        elif att_type == 'all_var':
            
            # [V B 2D]
            w_var = tf.get_variable('w_var', [h_dim*2, 1], initializer=tf.contrib.layers.xavier_initializer())
            b_var = tf.Variable( tf.random_normal([1]) )
            
            # ? bias nonlinear activation ?
            # [V B 1]
            var_weight = tf.sigmoid( tf.tensordot(h_temp, w_var, axes=1) + b_var )
            
            # [V B 2D] * [V B 1]
            h_weighted = h_temp * var_weight
            
            # [V B 2D]
            h_var_list = tf.split(h_weighted, num_or_size_splits = num_vari, axis = 0) 
            h_res = tf.squeeze(tf.concat(h_var_list, 2))
            
        elif att_type == 'sep_hidden':
            
            # [V B 2D]
            w_var = tf.get_variable('w_var', [h_dim*2, 1], initializer=tf.contrib.layers.xavier_initializer())
            b_var = tf.Variable( tf.random_normal([1]) )
            
            #[V B D]
            h_last, h_ctx = tf.split(h_temp, num_or_size_splits = 2, axis = 2)
            
            # ? bias nonlinear activation ?
            # [V B 1]
            var_weight = tf.sigmoid( tf.tensordot(h_temp, w_var, axes=1) )
            
            # [V B D]
            h_ctx_weighted = h_ctx * var_weight
            
            # [V B 2D]
            h_weighted = tf.concat([h_last, h_ctx_weighted], 2)
            
            # [V B 2D]
            h_var_list = tf.split(h_weighted, num_or_size_splits = num_vari, axis = 0) 
            h_res = tf.squeeze(tf.concat(h_var_list, 2))
            
        elif att_type == 'sep_tar':
            
            # [V B 2D]
            w_var = tf.get_variable('w_var', [h_dim*2, 1], initializer=tf.contrib.layers.xavier_initializer())
            b_var = tf.Variable( tf.random_normal([1]) )
            
            #[V-1 B 2D], [1 B 2D]
            h_indep, h_tar = tf.split(h_temp, num_or_size_splits = [num_vari - 1, 1], axis = 0)
            
            # ? bias nonlinear activation ?
            # [V-1 B 1]
            var_weight = tf.sigmoid( tf.tensordot(h_indep, w_var, axes=1) + b_var )
            
            # [V-1 B 2D]
            h_indep_weighted = h_indep * var_weight
            
            # [V B 2D]
            h_weighted = tf.concat([h_indep_weighted, h_tar], 0)
            
            # [V B 2D]
            h_var_list = tf.split(h_weighted, num_or_size_splits = num_vari, axis = 0) 
            h_res = tf.squeeze(tf.concat(h_var_list, 2)) 
         
        # consider both h_indep and h_tar to derive the weight 
        elif att_type == 'sep_tar_concat':
            
            # [V B 2D]
            w_var = tf.get_variable('w_var', [h_dim*4, 1], initializer=tf.contrib.layers.xavier_initializer())
            b_var = tf.Variable( tf.random_normal([1]) )
            
            #[V-1 B 2D], [1 B 2D]
            h_indep, h_tar = tf.split(h_temp, num_or_size_splits = [num_vari - 1, 1], axis = 0)
            #[V-1 B 2D]
            h_tar_tile = tf.tile(h_tar, [num_vari - 1, 1, 1])
            #[V-1 B 4D]
            h_concat = tf.concat( [h_indep, h_tar_tile], 2 )
            
            # ? bias nonlinear activation ?
            # [V-1 B 1]
            var_weight = tf.sigmoid( tf.tensordot(h_concat, w_var, axes=1) + b_var )
            
            # [V-1 B 2D]
            h_indep_weighted = h_indep * var_weight
            # [V B 2D]
            h_weighted = tf.concat([h_indep_weighted, h_tar], 0)
            
            # [V B 2D]
            h_var_list = tf.split(h_weighted, num_or_size_splits = num_vari, axis = 0) 
            h_res = tf.squeeze(tf.concat(h_var_list, 2))     
            
        else:
            print '[ERROR] variable attention type'
        
    return h_res, tf.nn.l2_loss(w_var), var_weight


# unified temporal attention 
def mv_attention_temp_weight_decay( h_list, h_dim, scope, step, step_idx, decay_activation, att_type ):
    
    with tf.variable_scope(scope):
        
        tmph = tf.stack(h_list, 0)
        # [V B T-1 D], [V, B, 1, D]
        tmph_before, tmph_last = tf.split(tmph, [step-1, 1], 2)
        
        # -- temporal logits
        if att_type == 'loc':
            
            w_temp = tf.get_variable('w_temp', [len(h_list), 1, 1, h_dim], initializer=tf.contrib.layers.xavier_initializer())
            # ?
            b_temp = tf.Variable( tf.zeros([len(h_list), 1, 1, 1]) )
            w_mul = tf.get_variable('w_mul', [len(h_list), 1, 1], initializer=tf.contrib.layers.xavier_initializer())
            
            # ? bias nonlinear activation ?
            #[V, B, T-1]
            temp_logit = tf.nn.tanh(tf.reduce_sum(tmph_before * w_temp + b_temp, 3))
            #temp_logit = tf.reduce_sum(tmph_before * w_temp, 3)
            
            # for decay
            # temp_logit = tf.sigmoid( tf.reduce_sum(tmph_before * w_temp + b_temp, 3) )
            
            # empty and relu activation 
            # ? use relu if with decay ?
            
        elif att_type == 'bilinear':
            
            w_temp = tf.get_variable('w_temp', [len(h_list), 1, h_dim, h_dim],\
                                     initializer=tf.contrib.layers.xavier_initializer())
            b_temp = tf.Variable( tf.zeros([len(h_list), 1, 1, 1]) )
            
            #[V, B, 1, D]
            tmp = tf.reduce_sum( tmph_last * w_temp, 3 )
            tmp = tf.expand_dims(tmp, 2)
        
            # ? bias nonlinear activation ?
            temp_logit = tf.reduce_sum(tmph_before * tmp, 3)
            
        elif att_type == 'concat':
            
            w_temp = tf.get_variable('w_temp', [len(h_list), 1, 1, h_dim*2],\
                                     initializer=tf.contrib.layers.xavier_initializer())
            b_temp = tf.Variable( tf.zeros([len(h_list), 1, 1, 1]) )
            
            # concatenate tmph_before and tmph_last
            # [V B T-1 D]
            last_tile = tf.tile(tmph_last, [1, 1, step-1, 1])
            tmph_tile = tf.concat( [tmph_before, last_tile], 3 )
            
            # ? bias nonlinear activation ?
            temp_logit = tf.reduce_sum( tmph_tile * w_temp, 3 ) 
            
        elif att_type == 'mlp':
            
            interm_dim = 8
            # [V 1 1 2D d]
            w_temp = tf.get_variable('w_temp', [len(h_list), 1, 1, h_dim*2, interm_dim],\
                                     initializer = tf.contrib.layers.xavier_initializer())
            b_temp = tf.Variable( tf.zeros([len(h_list), 1, 1, interm_dim]) )
            
            # concatenate tmph_before and tmph_last
            # [V B T-1 D]
            last_tile = tf.tile(tmph_last, [1, 1, step-1, 1])
            # [V B T-1 2D]
            tmph_tile = tf.concat( [tmph_before, last_tile], 3 )
            # [V B T-1 2D 1]
            tmph_tile = tf.expand_dims( tmph_tile, -1 )
            
            # [ V B T-1 d]
            # ? bias and activiation 
            interm_h = tf.nn.tanh( tf.reduce_sum( tmph_tile * w_temp + b_temp, 3 ) ) 
            
            # [V 1 1 d]
            w_mlp = tf.get_variable('w_mlp', [len(h_list), 1, 1, interm_dim],\
                                     initializer=tf.contrib.layers.xavier_initializer()) 
            b_mlp = tf.Variable( tf.zeros([len(h_list), 1, 1, 1]) )
            
            # ? bias nonlinear activation ?
            temp_logit = tf.sigmoid( tf.tf.reduce_sum( interm_h * w_mlp, 3 ) )
        
        else:
            print '[ERROR] attention type'
        
        
        # -- temporal decay
        # [V 1 ]
        w_decay = tf.get_variable('w_decay', [len(h_list), 1], initializer = tf.contrib.layers.xavier_initializer())
        w_decay = tf.square(w_decay)
        
        b_decay = tf.Variable( tf.zeros([len(h_list), 1]) )
        step_idx = tf.reshape(step_idx, [1, step-1])
        
        # new added
        # [V, T-1]
        v_step = tf.tile(step_idx, [len(h_list), 1])
        
        # ? add to regularization ?
        cutoff = tf.get_variable('cutoff', [len(h_list), 1], initializer = tf.contrib.layers.xavier_initializer())
        # [V, 1]
        cutoff_decay = tf.sigmoid(cutoff)*(step-1)
        
        # ? bias ?
        if decay_activation == 'exp':
            #temp_decay = tf.exp( tf.matmul(w_decay, -1*step_idx) )
            #temp_decay = tf.expand_dims(temp_decay, 1)
            
            # ? bias ?
            # [V 1] by [V T-1]
            temp_decay = tf.exp(w_decay*(cutoff_decay - v_step))
            # [V 1 T-1]
            temp_decay = tf.expand_dims(temp_decay, 1)
            
            
        elif decay_activation == 'sigmoid':
            #temp_decay = tf.sigmoid( tf.matmul(w_decay, -1*step_idx) )
            #temp_decay = tf.expand_dims(temp_decay, 1)
            
            # ? bias ?
            # [V 1] by [V T-1]
            temp_decay = tf.sigmoid(w_decay*(cutoff_decay - v_step)) 
            temp_decay = tf.expand_dims(temp_decay, 1)
            
        
        elif decay_activation == 'cutoff':
            
            # ? bias ?
            # [V 1] - [V T-1]
            temp_decay = tf.nn.relu( cutoff_decay - v_step ) 
            temp_decay = tf.expand_dims(temp_decay, 1)
            
        else:
            # no attention decay
            temp_weight = tf.nn.softmax( temp_logit )
            
            # temp_before [V B T-1 D], temp_weight [V B T-1]
            tmph_cxt = tf.reduce_sum( tmph_before*tf.expand_dims(temp_weight, -1), 2 )
            tmph_last = tf.squeeze( tmph_last, [2] ) 
            
            # [V B 2D]
            h_temp = tf.concat([tmph_last, tmph_cxt], 2)
            # ?
            #h_temp = tmph_last
            
            return h_temp, tf.nn.l2_loss(w_temp), temp_weight
        
        
        # -- decay on temporal logit
        # [V, B, T-1] * [V, 1, T-1]
        temp_logit_decay = temp_logit*temp_decay
        
        
        # -- attention weight
        if decay_activation == 'cutoff':
            # [V B 1]
            tmpsum = tf.expand_dims( tf.reduce_sum(temp_logit_decay, [2]), -1 )
            temp_weight = 1.0*temp_logit_decay/(tmpsum+1e-10)
            
        else:
            temp_weight = tf.nn.softmax( temp_logit_decay )
        
        
        # -- attention weighted context
        # tmph_before [V B T-1 D]
        tmph_cxt = tf.reduce_sum( tmph_before*tf.expand_dims(temp_weight, -1), 2 )
        
        # [context, last hidden]
        tmph_last = tf.squeeze( tmph_last, [2] )
        # [V B 2D]
        h_temp = tf.concat([tmph_last, tmph_cxt], 2)
        
        # ?
        #h_temp = tmph_last
        #h_var_list = tf.split(h_temp, num_or_size_splits = len(h_list), axis = 0) 
    
    #
    if decay_activation == 'cutoff':
        # ?
        return h_temp, tf.nn.l2_loss(w_temp) + tf.nn.l2_loss(cutoff), temp_weight
    
    else:
        return h_temp, [tf.nn.l2_loss(w_temp), tf.nn.l2_loss(w_decay)], temp_weight
#tf.squeeze(tf.concat(h_var_list, 2), [0])
    

def mv_dense( h_vari, dim_vari, scope, num_vari, dim_to, bool_output ):
    
    # argu [V B D]
    
    with tf.variable_scope(scope):
        
        # [V 1 D d]
        w = tf.get_variable('w', [ num_vari, 1, dim_vari, dim_to ], initializer=tf.contrib.layers.xavier_initializer())
        # [V 1 1 d]
        b = tf.Variable( tf.random_normal([ num_vari, 1, 1, dim_to ]) )
        
        # [V B D 1]
        h_expand = tf.expand_dims(h_vari, -1)
        
        # [V B D 1] * [V 1 D d] -> [V B d]
        # ?
        if bool_output == True:
            h = tf.reduce_sum(h_expand * w + b, 2)
        else:
            h = tf.nn.relu( tf.reduce_sum(h_expand * w + b, 2) ) 
    
    return h, tf.nn.l2_loss(w)    



# ---- multi-variate RNN ----

class tsLSTM_mv():
    
    def __init__(self, n_dense_dim_layers, n_lstm_dim_layers, n_steps, n_data_dim, session,\
                 lr, max_norm , n_batch_size, bool_residual, bool_att, temp_decay, temp_attention, \
                 l2_dense, l2_att, vari_attention):
        
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
        alpha = tf.constant( 0.3, dtype=tf.float32 )
        
        # begin to build the graph
        self.sess = session
        
        # residual connections
        if bool_residual == True:
            
            print ' [ERROR] Wrongly use residual connection'
            
        # no residual connection 
        else:
            
            # no attention
            if bool_att == '':
                
                print ' --- MV-RNN using no attention: '
                
                with tf.variable_scope('lstm'):
                    lstm_cell = MvLSTMCell( n_lstm_dim_layers[0], n_var = n_data_dim, \
                                            initializer=tf.contrib.layers.xavier_initializer() )
                    h, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = self.x, dtype = tf.float32)
            
                # obtain the last hidden state    
                h_temp = tf.transpose( h, [1,0,2] )
                h_last = h_temp[-1]
                
                h = h_last
               
                # ------------
                '''
                h_last_vari = tf.split( h_last, num_or_size_splits = self.N_DATA_DIM, axis = 1)
                
                # ? intermediate mv-dense ?
                #[V B D] - [V B D]
                #h_mv1, regu_mv_dense1 = mv_dense( h_last_vari, int(n_lstm_dim_layers[-1]/self.N_DATA_DIM), 'intermediate1',\
                #                            self.N_DATA_DIM,\
                #                            int(n_lstm_dim_layers[-1]/self.N_DATA_DIM) )
                
                # [V B D/2]
                h_mv, regu_mv_dense = mv_dense( h_last_vari, int(n_lstm_dim_layers[-1]/self.N_DATA_DIM), 'intermediate2',\
                                            self.N_DATA_DIM,\
                                            int(n_lstm_dim_layers[-1]/self.N_DATA_DIM)*2 )
                
                # reshape to [B H/2]
                h_var_list = tf.split(h_mv, num_or_size_splits = n_data_dim, axis = 0) 
                h = tf.squeeze(tf.concat(h_var_list, 2), [0])
                
                #n_dense_dim_layers[-1] = n_lstm_dim_layers[-1]/2
                '''
                # ------------
                
                # test
                #self.test = tf.shape(h)
                
                # ?
                h, regu_dense = plain_dense( h, n_lstm_dim_layers[-1], n_dense_dim_layers, 'dense', self.keep_prob )
                regu_all = l2_dense*regu_dense
                
                #regu_all = l2_dense*(regu_mv_dense)
                
            
            elif bool_att == 'temp':
                
                print ' --- MV-RNN using only temporal attention: '

                with tf.variable_scope('lstm'):
                    
                    lstm_cell = MvLSTMCell( n_lstm_dim_layers[0], n_var = n_data_dim ,\
                                           initializer = tf.contrib.layers.xavier_initializer() )
                    h, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = self.x, dtype = tf.float32)
                
                # [V B T D]
                h_list = tf.split(h, num_or_size_splits = self.N_DATA_DIM, axis = 2)
                
                
                # --- apply temporal attention 
                
                # [V B T D] - [V B 2D]
                # shape pof h_temp [V B 2D]
                h_temp, regu_att, self.att = \
                mv_attention_temp_weight_decay( h_list, int(n_lstm_dim_layers[-1]/self.N_DATA_DIM),\
                                                       'att', self.N_STEPS, steps, temp_decay, temp_attention )
                h = h_temp
                
                # ? intermediate mv-dense ?
                # [V B 2D] - [V B D]
                #h, regu_mv_dense = mv_dense( h_temp, 2*int(n_lstm_dim_layers[-1]/self.N_DATA_DIM), 'intermediate',\
                #                            self.N_DATA_DIM,\
                #                            int(n_lstm_dim_layers[-1]/self.N_DATA_DIM) )
               
                # test
                self.test = tf.shape(self.att)
                
                # reshape to [B 2H]
                h_var_list = tf.split(h, num_or_size_splits = n_data_dim, axis = 0) 
                h_att = tf.squeeze(tf.concat(h_var_list, 2), [0])
                
                # ---
                
                # ?
                h, regu_dense = plain_dense_leaky( h_att, 2*n_lstm_dim_layers[-1], n_dense_dim_layers, 'dense', \
                                                   self.keep_prob, alpha )
                # ?
                if temp_decay == '' or temp_decay == 'cutoff' :
                    regu_all = l2_dense*(regu_dense) + l2_att*regu_att
                else:
                    regu_all = l2_dense*regu_dense + l2_att*(regu_att[0] + regu_att[1]) 
                
            
            elif bool_att == 'vari':
                
                print ' --- MV-RNN using only variate attention: '
                
                with tf.variable_scope('lstm'):
                    lstm_cell = MvLSTMCell( n_lstm_dim_layers[0], n_var = n_data_dim ,\
                                           initializer = tf.contrib.layers.xavier_initializer() )
                    h, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = self.x, dtype = tf.float32)
                
                # [B T D*V] - [T B D*V]
                h_temp = tf.transpose(h, [1, 0, 2])
                # [B D*V]
                h_last = h_temp[-1]
                # [V B D]
                h_last_vari = tf.split(h_last, [int(n_lstm_dim_layers[0]/self.N_DATA_DIM)]*self.N_DATA_DIM, 1)
                
                
                # --- apply variate attention 
                
                # ? shape h_att [B 2H] or [B 2D]
                h_att, regu_att_vari, self.att = mv_attention_variate(h_last_vari,\
                                                                      int(n_lstm_dim_layers[0]/self.N_DATA_DIM),\
                                                                      'att_vari', self.N_DATA_DIM, vari_attention)
                self.test = tf.shape(self.att)
                # ---
                
                if vari_attention in [ 'sigmoid' ]:
                    h, regu_dense = plain_dense( h_att, n_lstm_dim_layers[-1], n_dense_dim_layers, 'dense', self.keep_prob )
                
                elif vari_attention in ['softmax', 'mlp']:
                    h, regu_dense = plain_dense( h_att, n_lstm_dim_layers[-1]/self.N_DATA_DIM*2, n_dense_dim_layers, 'dense',\
                                                self.keep_prob )
                else:
                    print '[ERROR] variate attention type'
                    
                # ?
                regu_all = l2_dense*regu_dense + l2_att*regu_att_vari
             
            
            elif bool_att == 'vari-mv':
                
                print ' --- MV-RNN using only variate attention with mv-dense: '
                
                with tf.variable_scope('lstm'):
                    lstm_cell = MvLSTMCell( n_lstm_dim_layers[0], n_var = n_data_dim ,\
                                           initializer = tf.contrib.layers.xavier_initializer() )
                    h, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = self.x, dtype = tf.float32)
                
                # [B T D*V]
                h_temp = tf.transpose(h, [1, 0, 2])
                # [B D*V]
                h_last = h_temp[-1]
                
                # [V B D]
                h_last_vari = tf.split(h_last, [int(n_lstm_dim_layers[0]/self.N_DATA_DIM)]*self.N_DATA_DIM, 1)
                
                # --- intermediate mv-dense
                # [V B D] - [V B d]
                
                interm_var_dim = int(n_lstm_dim_layers[-1]/self.N_DATA_DIM)*2
                
                h_mv, regu_mv_dense = mv_dense( h_last_vari, int(n_lstm_dim_layers[-1]/self.N_DATA_DIM), 'intermediate',\
                                               self.N_DATA_DIM, interm_dim_layer, False )
                
                # --- apply variate attention 
                
                # ? shape h_att [B 2H] or [B 2D]
                h_att, regu_att_vari, self.att = mv_attention_variate(h_mv, interm_var_dim,\
                                                                      'att_vari', self.N_DATA_DIM, vari_attention)
                
                # ---
                
                if vari_attention in [  'sigmoid' ]:
                    n_dense_dim_layers[-1] = interm_var_dim*self.N_DATA_DIM
                
                elif vari_attention == 'softmax':
                    n_dense_dim_layers[-1] = interm_var_dim*2
                    
                else:
                    print '[ERROR] variate attention type'
                
                
                # ?
                regu_all = l2_dense*regu_mv_dense + l2_att*regu_att_vari
                
            
            elif bool_att == 'vari-mv-output':
                
                print ' --- MV-RNN using only variate attention with mv-dense: '
                
                with tf.variable_scope('lstm'):
                    lstm_cell = MvLSTMCell( n_lstm_dim_layers[0], n_var = n_data_dim ,\
                                           initializer = tf.contrib.layers.xavier_initializer() )
                    h, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = self.x, dtype = tf.float32)
                
                # [B T D*V]
                h_temp = tf.transpose(h, [1, 0, 2])
                # [B D*V]
                h_last = h_temp[-1]
                
                
                #dropout
                h_last = tf.nn.dropout(h_last, self.keep_prob)
                
                
                # [V B D]
                h_last_vari = tf.split(h_last, [int(n_lstm_dim_layers[0]/self.N_DATA_DIM)]*self.N_DATA_DIM, 1)
                
                # --- intermediate mv-dense
                # [V B D] - [V B d]
                
                interm_var_dim = int(n_lstm_dim_layers[-1]/self.N_DATA_DIM)*2
                
                # h_mv [V B d]
                h_mv1, regu_mv_dense1 = mv_dense( h_last_vari, int(n_lstm_dim_layers[-1]/self.N_DATA_DIM), 'intermediate1',\
                                                self.N_DATA_DIM, interm_var_dim, False )
                
                
                h_mv, regu_mv_dense = mv_dense( h_mv1, interm_var_dim, 'intermediate',\
                                                self.N_DATA_DIM, 1, True )
                
                
                # --- derive variate attention 
                
                h_att, regu_att_vari, self.att = mv_attention_variate(h_last_vari,\
                                                                      int(n_lstm_dim_layers[0]/self.N_DATA_DIM),\
                                                                      'att_vari', self.N_DATA_DIM, vari_attention)
                self.test = tf.shape(self.att)
                
                # ? shape self.att [B V-1 1]
                #self.att, regu_att_vari = mv_attention_variate( h_last_vari, int(n_lstm_dim_layers[-1]/self.N_DATA_DIM),\
                #                                               'att_vari', self.N_DATA_DIM, vari_attention )
                
                
                # --- partial attention
                '''
                # [V-1 B d], [1 B d]
                h_mv_indep, h_mv_tar = tf.split(h_mv, [self.N_DATA_DIM-1, 1], 0)
                # [B V-1 d]
                h_mv_indep_trans = tf.transpose(h_mv_indep, [1, 0, 2])
                # [B, 1]
                h_mv_indep_weighted = tf.reduce_sum( h_mv_indep_trans*self.att, 1 )
                
                self.py = h_mv_indep_weighted + tf.squeeze(h_mv_tar, [0])
                '''
                # --- full attention 
                # [V B d] - [B V d]
                h_mv_trans = tf.transpose(h_mv, [1, 0, 2])
                # [B V d]*[B V 1]
                h_mv_weighted = tf.reduce_sum( h_mv_trans*self.att, 1 )
                
                self.py = h_mv_weighted
                
                
                # --- regularization
                regu_all = l2_dense*regu_mv_dense + l2_dense*regu_mv_dense1 + l2_att*regu_att_vari
                self.regularization = regu_all
                
                return 
            
            
            elif bool_att == 'both-att':
                
                print ' --- MV-RNN using both temporal and variate attention: ', bool_att
                
                with tf.variable_scope('lstm'):
                    lstm_cell = MvLSTMCell( n_lstm_dim_layers[0], n_var = n_data_dim ,\
                                           initializer = tf.contrib.layers.xavier_initializer() )
                    h, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = self.x, dtype = tf.float32)
                
                h_list = tf.split(h, [int(n_lstm_dim_layers[0]/self.N_DATA_DIM)]*self.N_DATA_DIM, 2)
                
                
                # --- apply temporal and variate attention 
                
                # shape h_temp [V B 2D]
                h_temp, regu_att_temp, self.att_temp = \
                mv_attention_temp_weight_decay( h_list, int(n_lstm_dim_layers[0]/self.N_DATA_DIM),\
                                                       'att_temp', self.N_STEPS, steps, temp_decay, temp_attention )
                
                # ? shape h_att [B 2H]
                h_att, regu_att_vari, self.att_vari = mv_attention_variate_temp(h_temp,\
                                                                                int(n_lstm_dim_layers[0]/self.N_DATA_DIM),\
                                                                                'att_vari', self.N_DATA_DIM, vari_attention)
                
                self.att = [self.att_temp, self.att_vari]
                
                # ---
                
                if vari_attention in [ 'all_var', 'sep_hidden', 'sep_tar', 'sep_tar_concat' ]:
                    h, regu_dense = plain_dense( h_att, n_lstm_dim_layers[-1]*2, n_dense_dim_layers, 'dense', self.keep_prob )
                
                elif vari_attention == 'sum':
                    h, regu_dense = plain_dense( h_att, n_lstm_dim_layers[-1]/self.N_DATA_DIM*4, n_dense_dim_layers, 'dense',\
                                                self.keep_prob )
                else:
                    print '[ERROR] variable attention type'
                    
                # ?
                regu_all = l2_dense*regu_dense + l2_att*(regu_att_temp + regu_att_vari)
                
                
            elif bool_att == 'both-pool':
                
                print ' --- MV-RNN using both temporal and variate attention: ', bool_att
                
                with tf.variable_scope('lstm'):
                    lstm_cell = MvLSTMCell( n_lstm_dim_layers[0], n_var = n_data_dim ,\
                                           initializer = tf.contrib.layers.xavier_initializer() )
                    h, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = self.x, dtype = tf.float32)
                
                h_list = tf.split(h, [int(n_lstm_dim_layers[0]/self.N_DATA_DIM)]*self.N_DATA_DIM, 2)
                
                
                # --- apply temporal and variate attention 
                
                # shape h_temp [V B 2D]
                h_temp = mv_pooling_temp( h_list, 'average',self.N_STEPS )
                
                # ? shape h_att [B 2H]
                h_att, regu_att_vari, self.att_vari = mv_attention_variate_temp(h_temp,\
                                                                                int(n_lstm_dim_layers[0]/self.N_DATA_DIM),\
                                                                                'att_vari', self.N_DATA_DIM, vari_attention)
                self.att = self.att_vari
                
                # ---
                
                if vari_attention in [ 'all_var', 'sep_hidden', 'sep_tar', 'sep_tar_concat' ]:
                    h, regu_dense = plain_dense( h_att, n_lstm_dim_layers[-1]*2, n_dense_dim_layers, 'dense', self.keep_prob )
                
                elif vari_attention == 'sum':
                    h, regu_dense = plain_dense( h_att, n_lstm_dim_layers[-1]/self.N_DATA_DIM*4, n_dense_dim_layers, 'dense',\
                                                self.keep_prob )
                else:
                    print '[ERROR] variable attention type'
                    
                # ?
                regu_all = l2_dense*regu_dense + l2_att*regu_att_vari
                
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
        
        # regularization in LSTM 
        self.regul_lstm = sum( tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() \
                                   if ("lstm" in tf_var.name and "input" in tf_var.name))
        
        # ?
        #self.regularization += self.regul_lstm

        
    def train_ini(self):  
        
        tmp_sq_diff = tf.square(self.y - self.py) 
        self.error_sum = tf.reduce_sum( tmp_sq_diff )
        self.error_mean = tf.reduce_mean( tmp_sq_diff )
        
        # ? loss function ? 
        self.loss = self.error_mean + self.regularization
        
        self.optimizer = \
        tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.loss)  
        
        self.init = tf.global_variables_initializer()
        self.sess.run( [self.init] )
        
    def train_batch(self, x_batch, y_batch, keep_prob ):
        
        _, tmp_loss, tmp_err_sum = self.sess.run([self.optimizer, self.loss, self.error_sum],\
                              feed_dict={self.x:x_batch, self.y:y_batch, self.keep_prob:keep_prob })
        
        return tmp_loss, tmp_err_sum

#   initialize inference         
    def inference_ini(self):
        
        # error metric
        self.rmse = tf.sqrt( tf.reduce_mean(tf.square(self.y - self.py)) )
        self.mae =  tf.reduce_mean( tf.abs(self.y - self.py) )
        self.mape = tf.reduce_mean( tf.abs( (self.y - self.py)*1.0/(self.y+1e-5) ) )
        
        
#   infer givn testing data    
    def inference(self, x_test, y_test, keep_prob):
        
        return self.sess.run([self.py, self.rmse, self.mae, self.mape], \
                             feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    def predict(self, x_test, y_test, keep_prob):
        
        return self.sess.run( self.py, feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    
    def test_regularization(self, x_test, y_test, keep_prob):
        return self.sess.run([self.regularization], feed_dict = {self.x:x_test, self.y:y_test,\
                                                                            self.keep_prob:keep_prob})
    
    def test_attention(self, x_test, y_test, keep_prob):
        return self.sess.run([self.att],  feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    def testfunc(self, x_batch, y_batch, keep_prob ):
        
        #tmpname= []
        #for tf_var in tf.trainable_variables():
        #    tmpname.append( tf_var.name )
            
        #self.test, regul_lstm    
        return self.sess.run([self.test],\
                            feed_dict={self.x:x_batch, self.y:y_batch, self.keep_prob:keep_prob })
  


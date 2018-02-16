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
from ts_mv_rnn_attention import *


    
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
                h_temp, regu_att, self.att = mv_attention_temp( h_list, int(n_lstm_dim_layers[-1]/self.N_DATA_DIM),\
                                                       'att', self.N_STEPS, steps, temp_decay, temp_attention )
               
                # test
                self.test = tf.shape(self.att)
                
                # reshape to [B 2H]
                h_var_list = tf.split(h_temp, num_or_size_splits = n_data_dim, axis = 0) 
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
                
            
            elif bool_att == 'vari-hidden':
                
                print ' --- MV-RNN using only variate attention on hiddens: '
                
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
             
            
            elif bool_att == 'vari-mix':
                
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
                
                # [V B T D]
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
  


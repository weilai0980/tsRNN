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
                
    return hiddens 

    
def res_dense(x, x_dim, hidden_dim, n_layers, scope, dropout_keep_prob):
    
        #dropout
        x = tf.nn.dropout(x, dropout_keep_prob)
        
        with tf.variable_scope(scope):
                # initilization
                w = tf.get_variable('w', [x_dim, hidden_dim], dtype = tf.float32,
                                          initializer = tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros([hidden_dim]))
                h = tf.nn.relu( tf.matmul(x, w) + b )

                regularization = tf.nn.l2_loss(w)
        #dropout
        h = tf.nn.dropout(h, dropout_keep_prob)
        
        for i in range(1, n_layers):
            
            with tf.variable_scope(scope+str(i)):
                w = tf.get_variable('w', [hidden_dim, hidden_dim],initializer=tf.contrib.layers.xavier_initializer())
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
        h = tf.nn.dropout(h, dropout_keep_prob)
        
        for i in range(1, len(dim_layers)):
            
            with tf.variable_scope(scope+str(i)):
                w = tf.get_variable('w', [dim_layers[i-1], dim_layers[i]],initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros( dim_layers[i] ))
                h = tf.nn.relu( tf.matmul(h, w) + b )
                
                regularization += tf.nn.l2_loss(w)
        
        return h, regularization

    
#---- Attention ----

# ref: a structured self attentive sentence embedding  
def attention_temporal( h, h_dim, att_dim, scope ):
    # tf.tensordot
    with tf.variable_scope(scope):
        
        w = tf.get_variable('w', [h_dim, att_dim], initializer=tf.contrib.layers.xavier_initializer())
        #? add bias?
        tmp_h = tf.nn.relu( tf.tensordot(h, w, axes=1) )

        w_logit = tf.get_variable('w_log', [att_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
        logit = tf.tensordot(tmp_h, w_logit, axes=1)
        
        alphas = tf.nn.softmax( tf.squeeze(logit) )
        
    return tf.reduce_sum(h*tf.expand_dims(alphas, -1), 1), alphas
    

# shape of h_list: [#variate, batch_size, steps, dim]
def attention_variate_softmax( h_list, h_dim_list, att_dim_list, att_dim_var, scope ):
    
    hh = []
    for i in range(len(h_list)):
        hh.append( attention_temporal(h_list[i], h_dim_list[i], att_dim_list[i], scope+str(i)) )
    #shape of hh: [#variate, batch_size, h_dim]
    
    v = []
    for i in range(len(h_list)):
        with tf.variable_scope(scope+'var'+str(i)):
            w = tf.get_variable('w', [h_dim_list[i], att_dim_var], initializer=tf.contrib.layers.xavier_initializer())
            #? add bias?
            v.append( tf.nn.relu(tf.matmul(hh[i], w)) )

    v = tf.stack(v, 0)
    v = tf.transpose(v, [1, 0, 2])
    # v shape [batch, variate, att_dim_var]
    
    w_att = tf.get_variable('w_', [att_dim_var, 1], initializer=tf.contrib.layers.xavier_initializer())
    logit = tf.tensordot(v, w_att, axes=1)
    alphas = tf.nn.softmax(logit)
    
    return attention_on_variate(alphas, h_list, h_dim_list)

def attention_on_variate(alpha, h_list, h_dim_list):
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
            
            if bool_att == True:
                h, self.att = attention_temporal( h, n_lstm_dim_layers[0], int(n_lstm_dim_layers[0]/2), 'att' )
            else:
                # obtain the last hidden state    
                tmp_hiddens = tf.transpose( h, [1,0,2]  )
                h = tmp_hiddens[-1]
            
            h, regul = res_dense( h, n_lstm_dim_layers[0], n_dense_dim_layers[0], len(n_dense_dim_layers), 'dense',\
                                  self.keep_prob )
        else:
            
            h, _ = plain_lstm( self.x, n_lstm_dim_layers, 'lstm')
            
            if bool_att==True:
                h, self.att = attention_temporal( h, n_lstm_dim_layers[-1], int(n_lstm_dim_layers[-1]/2), 'att' )
            else:
                # obtain the last hidden state
                tmp_hiddens = tf.transpose( h, [1,0,2]  )
                h = tmp_hiddens[-1]
            
            h, regul = plain_dense( h, n_lstm_dim_layers[-1], n_dense_dim_layers, 'dense', self.keep_prob )
            
        #dropout
        #last_hidden = tf.nn.dropout(last_hidden, self.keep_prob)
        with tf.variable_scope("output"):
            
            w = tf.get_variable('w', shape=[n_dense_dim_layers[-1], 1],\
                                     initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.zeros( [ 1 ] ))
            
            self.py = tf.matmul(h, w) + b
            self.regularization = regul + tf.nn.l2_loss(w)
            
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
        
        
#---- mulitvariate individual RNN ----

class tsLSTM_seperate_mv():
    
    def __init__(self, n_dense_dim_layers, n_lstm_dim_layers, n_steps, n_data_dim, session,\
                 lr, l2, max_norm , n_batch_size ):
        
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
            
            h, _  = res_lstm( current_x, n_lstm_dim_layers[0], len(n_lstm_dim_layers), 'lstm'+str(i) )
            # obtain the last hidden state    
            tmp_hiddens = tf.transpose( h, [1,0,2]  )
            h = tmp_hiddens[-1]
            
            concat_h.append(h)
         
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
            
        # TO DO: prediction merge, self attetion, mixture of expert
        

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
        
        _, c = self.sess.run([self.optimizer, self.cost],\
                                feed_dict={self.x:x_batch,\
                                           self.y:y_batch,\
                                           self.keep_prob:keep_prob\
                                 })
        return c

#   initialize inference         
    def inference_ini(self):

#       denormalzied RMSE  
        self.rmse = tf.sqrt( tf.reduce_mean( tf.square( self.y - self.py ) ) )
        
        
#   infer givn testing data    
    def inference(self, x_test, y_test, keep_prob):
        return self.sess.run([self.rmse], feed_dict={self.x:x_test,\
                                                self.y:y_test,\
                                                self.keep_prob:keep_prob\
                                                })
    
# ---- multi-variate RNN ----

class tsLSTM_mv():
    
    def __init__(self, n_dense_dim_layers, n_lstm_dim_layers, n_steps, n_data_dim, session,\
                 lr, l2, max_norm , n_batch_size, bool_residual ):
        
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
        
        if bool_residual:
            
            with tf.variable_scope('lstm'):
                lstm_cell = MvLSTMCell(120)
                #, initializer= tf.contrib.keras.initializers.glorot_normal()
                h, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = self.x, dtype = tf.float32)
            
            # obtain the last hidden state    
            tmp_hiddens = tf.transpose( h, [1,0,2]  )
            h = tmp_hiddens[-1]
            
            h, regul = res_dense( h, n_lstm_dim_layers[0], n_dense_dim_layers[0], len(n_dense_dim_layers), 'dense',\
                                  self.keep_prob )
            
        #dropout
        #last_hidden = tf.nn.dropout(last_hidden, self.keep_prob)
        with tf.variable_scope("output"):
            
            w = tf.get_variable('w', shape=[n_dense_dim_layers[-1], 1],\
                                     initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.zeros( [ 1 ] ))
            self.py = tf.matmul(h, w) + b
            
            
            self.regularization = regul + tf.nn.l2_loss(w)
            

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
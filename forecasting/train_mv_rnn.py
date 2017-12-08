#!/usr/bin/python
import sys

import collections
import hashlib
import numbers

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.rnn_cell_impl import *

# local 
from custom_rnn_cell import *
from ts_mv_rnn import *
from utils_libs import *


# ---- training process ----

if __name__ == '__main__':
    
    dataset_str = str(sys.argv[1])
    method_str = str(sys.argv[2])
    
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
    log_file   = "res/rnn"
    model_file = "res/model/rnn"

#   clean logs
    with open(log_file, "w") as text_file:
        text_file.close()

# --- network set-up ---
    
    # fixed
    para_input_dim = np.shape(xtrain)[-1]
    para_win_size =  np.shape(xtrain)[1]
    
    para_max_norm = 0.0
    para_is_stateful = False
    para_n_epoch = 500
    para_bool_residual = True
    para_bool_attention = True
    
    # plain
    # if residual layers are used, keep all dimensions the same 
    para_lstm_dims_plain = [96, 96, 96]
    para_dense_dims_plain = [32, 32, 32]

    para_lr_plain = 0.001
    para_batch_size_plain = 64
    
    para_l2_plain = 0.015
    para_keep_prob_plain = 1.0

    # seperate
    para_lstm_dims_sep = [96, 96, 96]
    para_dense_dims_sep = [32, 32, 32]

    para_lr_sep = 0.001
    para_batch_size_sep = 64
    
    para_l2_sep = 0.015
    para_keep_prob_sep = 0.8
    
    # mv
    para_lstm_dims_mv = [120, 120, 120]
    para_dense_dims_mv = [32, 32]

    para_lr_mv = 0.001
    para_batch_size_mv = 64
    
    para_l2_mv = 0.015
    para_keep_prob_mv = 0.8

    
#--- build and train the model ---
    
    tf.reset_default_graph()
    
    with tf.Session() as sess:
        
        if method_str == 'rnn':
            reg = tsLSTM_plain(para_dense_dims_plain, para_lstm_dims_plain, \
                                    para_win_size, para_input_dim, sess, \
                                    para_lr_plain, para_l2_plain, para_max_norm, para_batch_size_plain, \
                                    para_bool_residual, para_bool_attention)
            
            log_file += "_plain.txt"
            model_file += "_plain-{epoch:02d}.hdf5"
            
            para_batch_size = para_batch_size_plain
            para_keep_prob = para_keep_prob_plain
            
        elif method_str == 'sep':
            reg = tsLSTM_seperate_mv(para_dense_dims, para_lstm_dims, \
                                 para_win_size,   para_input_dim, sess, \
                                 para_lr, para_l2,para_max_norm, para_batch_size)
            log_file += "_sep.txt"
            model_file += "_sep-{epoch:02d}.hdf5"
            
            para_batch_size = para_batch_size_sep
            para_keep_prob = para_keep_prob_sep
        
        elif method_str == 'mv':
            reg = tsLSTM_mv(para_dense_dims, para_lstm_dims, \
                                 para_win_size,   para_input_dim, sess, \
                                 para_lr, para_l2,para_max_norm, para_batch_size, para_bool_residual)
            log_file += "_mv.txt"
            model_file += "_mv-{epoch:02d}.hdf5"
            
            para_batch_size = para_batch_size_mv
            para_keep_prob = para_keep_prob_mv
        
        
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
            
            print reg.test_attention(xtest[:2], ytest[:2],  para_keep_prob)
            
            with open(log_file, "a") as text_file:
                    text_file.write("At epoch %d: loss %f, train %f, test %f\n" % ( epoch, tmpc*1.0/total_iter, \
                                                                                   tmp_train_acc[0], tmp_test_acc[0] ))  
        
        
        print "Optimization Finished!"
        
        
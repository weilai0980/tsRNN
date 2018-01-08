#!/usr/bin/python
import sys

import collections
import hashlib
import numbers

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.rnn_cell_impl import *

import pickle

# local 
from custom_rnn_cell import *
from ts_mv_rnn import *
from utils_libs import *


# check list:
# attention: non, temp, both
# lr and l2
# bias and activation in attention
# regularization on attention
# regularization on lstm

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
    attention_file = "res/att"
    PIK = "epoch_err_"
    
# --- network set-up ---
    
    # fixed
    para_input_dim = np.shape(xtrain)[-1]
    para_win_size =  np.shape(xtrain)[1]
    
    para_max_norm = 0.0
    para_is_stateful = False
    para_n_epoch = 500
    
    # if residual layers are used, keep all dimensions the same 
    para_bool_residual = False
    para_bool_attention = 'temp'
    
    # -- plain --
    para_lstm_dims_plain = [96]
    #[96, 96, 96]
    para_dense_dims_plain = [32, 16, 8]
    #[32, 32, 32]

    para_lr_plain = 0.002
    #0.002
    para_batch_size_plain = 64
    
    para_l2_plain = 0.01
    #0.01
    para_keep_prob_plain = 1.0

    # -- seperate --
    para_lstm_dims_sep = [96]
    #[96, 96, 96]
    para_dense_dims_sep = [32, 16, 8]
    #[32, 32, 32]

    para_lr_sep = 0.002
    #0.002
    para_batch_size_sep = 64
    
    para_l2_sep = 0.02
    #0.02
    para_keep_prob_sep = 1.0
    
    # -- mv --
    para_lstm_dims_mv = [120]
    para_dense_dims_mv = [32, 16, 8]
    # no att: 32, 16, 8

    para_lr_mv = 0.002
    para_batch_size_mv = 64
    
    para_l2_mv = 0.01
    # no att: 0.01
    # temp att:
    # temp-var att: 
    para_keep_prob_mv = 1.0
    
    
#--- retrieve the trained model ---    
    
    
#--- build and train the model ---
    
    # clear graph
    tf.reset_default_graph()
    
    with tf.Session() as sess:
        
        if method_str == 'plain':
            '''
            reg = tsLSTM_plain(para_dense_dims_plain, para_lstm_dims_plain, \
                                    para_win_size, para_input_dim, sess, \
                                    para_lr_plain, para_l2_plain, para_max_norm, para_batch_size_plain, \
                                    para_bool_residual, para_bool_attention)
            '''
            
            log_file += "_plain.txt"
            model_file += "_plain"
            attention_file += "_plain.txt"
            PIK += "plain.dat"
            
            para_batch_size = para_batch_size_plain
            para_keep_prob = para_keep_prob_plain
            
        elif method_str == 'sep':
            '''
            reg = tsLSTM_seperate(para_dense_dims_sep, para_lstm_dims_sep, \
                                     para_win_size, para_input_dim, sess, \
                                     para_lr_sep, para_l2_sep, para_max_norm, para_batch_size_sep, \
                                     para_bool_residual, para_bool_attention)
            '''
            
            log_file += "_sep.txt"
            model_file += "_sep"
            attention_file += "_sep.txt"
            PIK += "sep.dat"
            
            para_batch_size = para_batch_size_sep
            para_keep_prob = para_keep_prob_sep
        
        elif method_str == 'mv':
            
            reg = tsLSTM_mv(para_dense_dims_mv, para_lstm_dims_mv, \
                            para_win_size,   para_input_dim, sess, \
                            para_lr_mv, para_l2_mv, para_max_norm, para_batch_size_mv, para_bool_residual, para_bool_attention)
            
            
            log_file += "_mv.txt"
            model_file += "_mv"
            attention_file += "_mv.txt"
            PIK += "mv.dat"
            
            para_batch_size = para_batch_size_mv
            para_keep_prob = para_keep_prob_mv
        
        '''
        # initialize the network
        reg.train_ini()
        reg.inference_ini()
        '''
        
        total_cnt   = np.shape(xtrain)[0]
        total_iter = int(total_cnt/para_batch_size)
        total_idx = range(total_cnt)
        
        
        # find the best model 
        with open(PIK, "rb") as f:
            epoch_tr_ts = pickle.load(f)
        
        # for test
        print len(epoch_tr_ts)
        
        best_epoch = min(epoch_tr_ts, key = lambda x:x[2] )[0]
        best_model_file = model_file + "_" + str(best_epoch) + ".ckpt"
        
        print '-- best epoch: ', best_epoch
        
        # model restoring 
        #saver = tf.train.Saver()        
        
        #saver.restore(sess, best_model_file)
        
        # use the best model to evalute 
        
        '''
        tmp_test_acc  = reg.inference(xtest, ytest,  para_keep_prob) 
        tmp_train_acc = reg.inference(xtrain,ytrain, para_keep_prob)

            # monitor training indicators
            print "At epoch %d: loss %f, train %s, test %s " % ( epoch, tmpc*1.0/total_iter, \
                                                                  tmp_train_acc, tmp_test_acc ) 
            

            # write performance indicators to txt files
            if para_bool_attention == 'both' or para_bool_attention == 'temp' :
                with open(attention_file, "a") as text_file:
                    text_file.write("-- At epoch %d: %s \n" % (epoch, \
                                                           str(reg.test_attention(xtest[:1], ytest[:1], para_keep_prob)))) 
            
            with open(log_file, "a") as text_file:
                text_file.write("At epoch %d: loss %f, train %f, test %f\n" % ( epoch, tmpc*1.0/total_iter, \
                                                                                   tmp_train_acc[0], tmp_test_acc[0] ))
                
        
        print "Evaluation Finished!"
        '''

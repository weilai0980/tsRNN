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

# fix the random seed to stabilize the network 
#np.random.seed(1)
#tf.set_random_seed(1)

# ---- check list ----
# attention: non, temp, both
# lr and l2

# bias and activation in attention

# regularization on attention
# regularization on lstm

# ---- Load data ----
    
dataset_str = str(sys.argv[1])
method_str = str(sys.argv[2])
    
file_dic = {}
    
file_addr = ["../../dataset/dataset_ts/air_xtrain.dat", \
                 "../../dataset/dataset_ts/air_xtest.dat",\
                 "../../dataset/dataset_ts/air_ytrain.dat", \
                 "../../dataset/dataset_ts/air_ytest.dat"]
file_dic.update( {"air": file_addr} )
    
file_addr = ["../../dataset/dataset_ts/energy_xtrain.dat", \
                 "../../dataset/dataset_ts/energy_xtest.dat",\
                 "../../dataset/dataset_ts/energy_ytrain.dat", \
                 "../../dataset/dataset_ts/energy_ytest.dat"]
file_dic.update( {"energy": file_addr} )
    
file_addr = ["../../dataset/dataset_ts/pm25_xtrain.dat", \
                 "../../dataset/dataset_ts/pm25_xtest.dat",\
                 "../../dataset/dataset_ts/pm25_ytrain.dat", \
                 "../../dataset/dataset_ts/pm25_ytest.dat"]
file_dic.update( {"pm25": file_addr} )
    
file_addr = ["../../dataset/dataset_ts/plant_xtrain.dat", \
                 "../../dataset/dataset_ts/plant_xtest.dat",\
                 "../../dataset/dataset_ts/plant_ytrain.dat", \
                 "../../dataset/dataset_ts/plant_ytest.dat"]
file_dic.update( {"plant": file_addr} )
     
print "Loading file at", file_dic[dataset_str][0] 
files_list = file_dic[dataset_str]
    
    
# ---- format and reshape data ---- 

# !! including feature normalization    
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
    

# ---- training log set-up ---- 
log_file   = "res/rnn"
model_file = "res/model/rnn"
attention_file = "res/att"
pre_file = "res/pred/pred_"
PIK = "epoch_err_"
    
epoch_tr_ts_err = []
    
# ---- network set-up ----
    
# fixed
para_input_dim = np.shape(xtrain)[-1]
para_win_size =  np.shape(xtrain)[1]
    
para_max_norm = 0.0
para_is_stateful = False
para_n_epoch = 100
    
# if residual layers are used, keep all dimensions the same 
para_bool_residual = False
para_bool_attention = 'vari-mv-output'
# temp, var, var-pre, both-att, both-pool, vari-mv-output'
    
# -- plain --
    
para_lstm_dims_plain = [64]
para_dense_dims_plain = [32, 16, 8]

para_lr_plain = 0.001
#0.002
para_batch_size_plain = 64
    
#para_l2_plain = 0.0001
#0.01
#para_keep_prob_plain = 1.0

    
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
    
para_lstm_dims_mv = [70]
para_dense_dims_mv = [32, 8]
#[32, 16, 8]

para_lr_mv = 0.002
# plant 0.002
# energy 0.003
para_batch_size_mv = 64
    
#para_l2_dense_mv = 0.1
para_l2_att_mv = 0.00001
    # no att: 0.001
    # temp loc : 0.001, 0.00001, 161 
    # temp loc cutoff:  0.03, 0.00001, 161
    # temp loc vari sep_tar:
#para_keep_prob_mv = 1.0
    
    # attention types
para_temp_decay_type = ''
    # cutoff
para_temp_attention_type = ''
    # loc, concate
para_vari_attention_type = 'softmax-all'
    # concat, sum, 'all_var', sep_tar
para_pool_type = ''
    # max, average, 
# ------------------------------------------------



def train_nn( l2_dense, dropout_keep_prob, log_file, pre_file ):   
    
# ---- build and train the model ----
    
    # clear graph
    tf.reset_default_graph()
    
    # fix the random seed to stabilize the network 
    #np.random.seed(1)
    #tf.set_random_seed(1)
    
    with tf.Session() as sess:
        
        # fix the random seed to stabilize the network 
        #np.random.seed(1)
        #tf.set_random_seed(1)
        
        if method_str == 'plain':
            
            print ' ---', method_str, ' parameter: ', l2_dense, dropout_keep_prob
            
            reg = tsLSTM_plain(para_dense_dims_plain, para_lstm_dims_plain, \
                                    para_win_size, para_input_dim, sess, \
                                    para_lr_plain, l2_dense, para_max_norm, para_batch_size_plain, \
                                    para_bool_residual, para_bool_attention)
            
            log_file += "_plain.txt"
            #model_file += "_plain"
            #attention_file += "_plain.txt"
            #PIK += "plain.dat"
            
            para_batch_size = para_batch_size_plain
            para_keep_prob = dropout_keep_prob
            
        elif method_str == 'sep':
            
            print ' ---', method_str, ' parameter: ', l2_dense, dropout_keep_prob
            
            reg = tsLSTM_seperate(para_dense_dims_sep, para_lstm_dims_sep, \
                                     para_win_size, para_input_dim, sess, \
                                     para_lr_sep, l2_dense, para_max_norm, para_batch_size_sep, \
                                     para_bool_residual, para_bool_attention)
            
            log_file += "_sep.txt"
            #model_file += "_sep"
            #attention_file += "_sep.txt"
            #PIK += "sep.dat"
            
            para_batch_size = para_batch_size_sep
            para_keep_prob = dropout_keep_prob
        
        elif method_str == 'mv':
            
            print ' ---', method_str, ' parameter: ', l2_dense, dropout_keep_prob
            
            reg = tsLSTM_mv(para_dense_dims_mv, para_lstm_dims_mv, \
                            para_win_size,   para_input_dim, sess, \
                            para_lr_mv, para_max_norm, para_batch_size_mv, para_bool_residual, \
                            para_bool_attention, para_temp_decay_type, para_temp_attention_type, l2_dense,\
                            para_l2_att_mv, para_vari_attention_type)
            
            
            log_file += "_mv.txt"
            #model_file += "_mv"
            #attention_file += "_mv.txt"
            #PIK += "mv.dat"
            
            para_batch_size = para_batch_size_mv
            para_keep_prob = dropout_keep_prob
        
        #   clean logs
        with open(log_file, "w") as text_file:
            text_file.close()
            
        #with open(attention_file, "w") as text_file:
        #    text_file.close()
        
        # initialize the network
        reg.train_ini()
        reg.inference_ini()
        
        # perpare for data shuffling
        total_cnt  = np.shape(xtrain)[0]
        total_iter = int(total_cnt/para_batch_size)
        total_idx = range(total_cnt)
        
        # test
        #print '? ? ? :', reg.testfunc(xtest, ytest, para_keep_prob)
        
        # set up model saver
        saver = tf.train.Saver(max_to_keep = para_n_epoch)
        
        # training epoches 
        for epoch in range(para_n_epoch):
            
            loss_epoch = 0.0
            err_sum_epoch = 0.0
            
            np.random.shuffle(total_idx)
            for i in range(total_iter):
                
                # shuffle training data
                batch_idx = total_idx[ i*para_batch_size: (i+1)*para_batch_size ] 
                batch_x = xtrain[ batch_idx ]
                batch_y = ytrain[ batch_idx ]            
                
                tmp_loss, tmp_err = reg.train_batch(batch_x, batch_y, para_keep_prob)
                
                loss_epoch += tmp_loss
                err_sum_epoch += tmp_err
                
            # [self.y_hat, self.rmse, self.mae, self.mape]
            yh_test, test_rmse_epoch, test_mae_epoch, test_mape_epoch = reg.inference(xtest, ytest, para_keep_prob)
            train_rmse_epoch = sqrt(1.0*err_sum_epoch/total_cnt)

            # monitor training indicators
            print "At epoch %d: loss %f, train %s, test %s, %f, %f " % ( epoch, loss_epoch*1.0/total_iter,\
                                                                        train_rmse_epoch, test_rmse_epoch, \
                                                                        test_mae_epoch,   test_mape_epoch ) 
            
            
            print np.squeeze(yh_test)[:5], np.squeeze(ytest)[:5]
            
            #if method_str == 'mv':
            #    print 'regular: ', reg.test_regularization(xtest, ytest,  para_keep_prob) 

            print '\n'
            
            
            # ---- logs of epoch performance    
            
            # write attention weights to txt files
            #if para_bool_attention != '' :
            #    with open(attention_file, "a") as text_file:
            #        text_file.write("-- At epoch %d: %s \n" % (epoch, \
            #                                               (reg.test_attention(xtest[:1000], ytest[:1000], para_keep_prob)))) 
            
            # write training and testing errors to txt files
            with open(log_file, "a") as text_file:
                text_file.write("At epoch %d: loss %f, train %f, test %f, %f, %f \n" % (epoch, loss_epoch*1.0/total_iter, \
                                                                               train_rmse_epoch, test_rmse_epoch,\
                                                                               test_mae_epoch,   test_mape_epoch) )
            
            
            # save epoch errors and the model
            #epoch_tr_ts_err.append([epoch, sqrt(1.0*err_sum_epoch/total_cnt), test_error_epoch[0]])
            #with open(PIK, "wb") as f:
            #    pickle.dump(epoch_tr_ts_err, f)
                
            #save_path = saver.save(sess, model_file + "_" + str(epoch) + ".ckpt")
            
            
            
            # save testing true, prediction values and attention
            
            if para_bool_attention =='':
                
                y_yh = np.concatenate( [ytest, yh_test], 1 )
                y_yh.dump(pre_file + str(epoch) + ".dat")
            
            else:
                tmp_att = reg.test_attention(xtest, ytest, para_keep_prob)
                
                print np.shape(tmp_att)
                tmp_att = np.squeeze(tmp_att)
                
                y_yh = np.concatenate( [ytest, yh_test], 1 )
                y_yh_att = np.concatenate( [y_yh, tmp_att], 1 )
            
                y_yh_att.dump(pre_file + str(epoch) + ".dat")
            
                print tmp_att[:1]
            
            
            #with open( pre_file + str(epoch) + ".txt", "w" ) as text_file:
            #    text_file.close()
            
            #with open( pre_file + str(epoch) + ".txt", "w" ) as text_file:
            #    text_file.write( "%s\n" %( str(list(zip(np.squeeze(ytest), np.squeeze(yh_test)))) ) )
            
            
            # ---
        
        print "Optimization Finished!"

    
# ----- loop over dropout and l2 

for tmp_keep_prob in [1.0, 0.6]:
    for tmp_l2 in [0.0001, 0.001, 0.01, 0.1, 1, 2]:
        log_file   = "res/rnn_" + str(tmp_keep_prob) + str(tmp_l2) 
            
        pre_file = "res/pred/pred_" + str(tmp_keep_prob) + str(tmp_l2) + "_"
            
        train_nn( tmp_l2, tmp_keep_prob, log_file, pre_file ) 
        
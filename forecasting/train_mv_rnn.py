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
from ts_clstm import *
from utils_libs import *

# fix the random seed to stabilize the network 
#np.random.seed(1)
#tf.set_random_seed(1)

# ------ check list ------
# attention: non, temp, both
# lr and l2

# bias and activation in attention

# regularization on attention
# regularization on lstm

# ------ Load data ------
    
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

file_addr = ["../../dataset/dataset_ts/temp_xtrain.dat", \
             "../../dataset/dataset_ts/temp_xtest.dat",\
             "../../dataset/dataset_ts/temp_ytrain.dat", \
             "../../dataset/dataset_ts/temp_ytest.dat"]
file_dic.update( {"temp": file_addr} )

file_addr = ["../../dataset/dataset_ts/syn_xtrain.dat", \
             "../../dataset/dataset_ts/syn_xtest.dat",\
             "../../dataset/dataset_ts/syn_ytrain.dat", \
             "../../dataset/dataset_ts/syn_ytest.dat"]
file_dic.update( {"syn": file_addr} )
     
print "Loading file at", file_dic[dataset_str][0] 
files_list = file_dic[dataset_str]
    
    
# ------ format and reshape data ------ 

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

# align the dimension    
ytrain = np.expand_dims( ytrain, 1 ) 
ytest  = np.expand_dims( ytest,  1 )

print np.shape(xtrain), np.shape(ytrain), np.shape(xtest), np.shape(ytest)
    

# ------ training log set-up ------ 
log_file   = "res/rnn"
model_file = "res/model/rnn"
attention_file = "res/att"
pre_file = "res/pred/pred_"
PIK = "epoch_err_"
    
epoch_tr_ts_err = []
    
# ------ network set-up ------
    
# fixed
para_input_dim = np.shape(xtrain)[-1]
para_win_size =  np.shape(xtrain)[1]

para_is_stateful = False
para_n_epoch = 60
    
# if residual layers are used, keep all dimensions the same 
para_bool_residual = False


# -- clstm --
    
para_attention_clstm = ''
para_lstm_dims_clstm = [256]
para_dense_dims_clstm = [128, 64, 8]

para_lr_clstm = 0.002
para_batch_size_clstm = 64


# -- plain --
    
para_attention_plain = 'temp'
para_lstm_dims_plain = [256]
para_dense_dims_plain = [128, 64, 8]

para_lr_plain = 0.001
para_batch_size_plain = 64
    
# -- mv --

para_attention_mv = 'both-att'
# temp, var, var-pre, both-att, both-pool, vari-mv-output'

hidden_dim_dic = {}
hidden_dim_dic.update( {"energy": [280]} )
hidden_dim_dic.update( {"plant": [160]} )
hidden_dim_dic.update( {"pm25": [175]} )
hidden_dim_dic.update( {"syn": [220]} ) 

lr_dic = {}
lr_dic.update( {"energy": 0.001} )
lr_dic.update( {"plant": 0.001} )
lr_dic.update( {"pm25": 0.003} )
lr_dic.update( {"syn": 0.001} ) 

maxnorm_dic = {}
maxnorm_dic.update( {"energy": 3.0} )
maxnorm_dic.update( {"plant": 3.0} )
maxnorm_dic.update( {"pm25": 4.0} )
maxnorm_dic.update( {"syn": 4.0} ) 

# dropout regularization    
#para_max_norm = 0.0
#para_keep_prob_mv = [1.0, 1.0, 1.0]

# norm regularization
para_dense_regul_type_mv= 'l2'
# l1, l2
para_l2_att_mv = 0.00001

# layer normalization
para_layer_norm = ''

para_lstm_dims_mv = hidden_dim_dic[dataset_str] 
# temp 17
# energy 14: 210, lr 0.005, 
# plant 8 : epoch 60, lr 0,003, 3.0, [0.5, 0.8, 1.0]
# pm25 7 : lr 0.001, 175, 3.0, [0.5, 0.5, 1.0]
# synthetic  11: lr 0.001
para_dense_dims_mv = [ ]

para_lr_mv = lr_dic[dataset_str]
para_batch_size_mv = 64
    

para_temp_attention_type = 'temp_loc'
# loc, concate
para_temp_decay_type = ''
# cutoff
para_vari_attention_type = 'vari_softmax_all'
para_pool_type = ''
para_loss_type = 'mse'
# mse,
# lk: likelihood 
para_loss_granularity = ''
# step, last

# -- seperate --

# per variable
para_lstm_dims_sep = [i/para_input_dim for i in hidden_dim_dic[dataset_str]]
para_dense_dims_sep = [ ]

para_lr_sep = lr_dic[dataset_str]
para_batch_size_sep = 64
    
# ------------------------------------------------

def train_nn( l2_dense, dropout_keep_prob, log_file, pre_file ):   
    
    # ---- build and train the model ----
    
    # clear graph
    tf.reset_default_graph()
    
    # ---- fix the random seed to stabilize the network 
    # np.random.seed(1)
    # tf.set_random_seed(1)
    
    with tf.Session() as sess:
        
        # fix the random seed to stabilize the network 
        # np.random.seed(1)
        # tf.set_random_seed(1)
        
        if method_str == 'plain':
            
            # ?
            if dropout_keep_prob == 1.0:
                
                para_keep_prob = [1.0, 1.0, 1.0]
                para_max_norm = maxnorm_dic[dataset_str]
            
            elif dropout_keep_prob == 0.8:
                
                para_keep_prob = [dropout_keep_prob, 1.0, 1.0]
                # ?
                para_max_norm = maxnorm_dic[dataset_str]
            
            else:
                
                para_keep_prob = [dropout_keep_prob, dropout_keep_prob + 0.3, 1.0]
                para_max_norm = maxnorm_dic[dataset_str]
                
            
            print ' ---', method_str, ' parameter: ', ' l2-', l2_dense, ' dropout-', para_keep_prob, \
            ' maxnorm-', para_max_norm
            
            reg = tsLSTM_plain(para_dense_dims_plain, para_lstm_dims_plain, \
                               para_win_size, para_input_dim, sess, \
                               para_lr_plain, l2_dense, para_max_norm, \
                               para_batch_size_plain, para_bool_residual, \
                               para_attention_plain, para_l2_att_mv)
            
            log_file += "_plain.txt"
            #model_file += "_plain"
            #attention_file += "_plain.txt"
            #PIK += "plain.dat"
            
            para_batch_size = para_batch_size_plain
            para_attention = para_attention_plain
            
        elif method_str == 'sep':
            
            # ?
            if dropout_keep_prob == 1.0:
                
                para_keep_prob = [1.0, 1.0, 1.0]
                para_max_norm = 0.0
            
            elif dropout_keep_prob == 0.8:
                
                para_keep_prob = [0.8, 0.8, 1.0]
                # ?
                para_max_norm = maxnorm_dic[dataset_str]
                
            else:
                
                para_keep_prob = [dropout_keep_prob, dropout_keep_prob + 0.3, 1.0]
                # ?
                para_max_norm = maxnorm_dic[dataset_str]
            
            
            print ' ---', method_str, ' parameter: ', ' l2-', l2_dense, ' dropout-', para_keep_prob, \
            ' maxnorm-', para_max_norm
            
            reg = tsLSTM_seperate(para_dense_dims_sep, para_lstm_dims_sep, \
                                  para_win_size, para_input_dim, sess, \
                                  para_lr_sep, l2_dense, 0.0, para_batch_size_sep, \
                                  para_bool_residual, para_attention_mv,\
                                  para_temp_attention_type, para_vari_attention_type,\
                                  para_dense_regul_type_mv, para_l2_att_mv)
            
            log_file += "_sep.txt"
            #model_file += "_sep"
            #attention_file += "_sep.txt"
            #PIK += "sep.dat"
            
            para_batch_size = para_batch_size_sep
            para_attention = para_attention_plain
        
        
        elif method_str == 'mv':
            
            # ?
            if dropout_keep_prob == 1.0:
                
                para_keep_prob = [1.0, 1.0, 1.0]
                para_max_norm = 0.0
            
            elif dropout_keep_prob == 0.8:
                
                para_keep_prob = [0.8, 0.8, 1.0]
                # ?
                para_max_norm = maxnorm_dic[dataset_str]
                
            else:
                
                para_keep_prob = [dropout_keep_prob, dropout_keep_prob + 0.3, 1.0]
                # ?
                para_max_norm = maxnorm_dic[dataset_str]
            
            
            print ' ---', method_str, ' parameter: ', ' l2-', l2_dense, ' dropout-', para_keep_prob, \
            ' maxnorm-', para_max_norm
            
            reg = tsLSTM_mv(para_dense_dims_mv, para_lstm_dims_mv, para_win_size, para_input_dim, sess, \
                            para_lr_mv, para_max_norm, para_batch_size_mv, para_bool_residual,\
                            para_attention_mv, para_temp_decay_type, para_temp_attention_type,\
                            l2_dense, para_l2_att_mv, para_vari_attention_type,\
                            para_loss_type, para_dense_regul_type_mv,\
                            para_layer_norm)
            
            log_file += "_mv.txt"
            #model_file += "_mv"
            #attention_file += "_mv.txt"
            #PIK += "mv.dat"
            
            para_batch_size = para_batch_size_mv
            para_attention = para_attention_mv
            
        elif method_str == 'clstm':
            
            # ?
            if dropout_keep_prob == 1.0:
                
                para_keep_prob = [1.0, 1.0, 1.0]
                para_max_norm = maxnorm_dic[dataset_str]
            
            elif dropout_keep_prob == 0.8:
                
                para_keep_prob = [dropout_keep_prob, 1.0, 1.0]
                # ?
                para_max_norm = maxnorm_dic[dataset_str]
            
            else:
                
                para_keep_prob = [dropout_keep_prob, dropout_keep_prob + 0.3, 1.0]
                para_max_norm = maxnorm_dic[dataset_str]
                
            print ' ---', method_str, ' parameter: ', ' l2-', l2_dense, ' dropout-', para_keep_prob, \
            ' maxnorm-', para_max_norm
            
            reg = cLSTM_causal(para_dense_dims_clstm, para_lstm_dims_clstm, \
                               para_win_size, para_input_dim, sess, \
                               para_lr_clstm, l2_dense, para_max_norm, \
                               para_batch_size_clstm, para_bool_residual, \
                               para_attention_clstm, para_l2_att_mv, l2_dense)
            
            log_file += "_clstm.txt"
            #model_file += "_plain"
            #attention_file += "_plain.txt"
            #PIK += "plain.dat"
            
            para_batch_size = para_batch_size_plain
            para_attention = ''
            
            # .... test ....
            #reg.train_ini()
            
            #print "-------------------", reg.collect_coeff_values('lstm')
            
            #return 
            # ... ......... 
        
        
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
            
            # epoch: learning rate update
            # if epoch >= 20:
            #    reg.train_update_optimizer(para_lr_mv)
            
            # each epoch, performance record 
            if para_attention =='':
                
                # [self.y_hat, self.rmse, self.mae, self.mape]
                yh_test, test_rmse_epoch, test_mae_epoch, test_mape_epoch = reg.inference(xtest, ytest, [1.0, 1.0])
                train_rmse_epoch = sqrt(1.0*err_sum_epoch/total_cnt)
                
                if method_str == 'mv': 
                    # dump file of truth and prediction
                    y_yh = np.concatenate( [ytest, yh_test], 1 )
                    y_yh.dump(pre_file + str(epoch) + ".dat")
                    
                elif method_str == 'clstm':
                    
                    regularized_weight = reg.collect_weight_values()
                    #regularized_weight.dump(pre_file + str(epoch) + '_reg_w' + ".dat")
                    
            else:
                
                if method_str == 'mv':
                    # [self.y_hat, self.rmse, self.mae, self.mape]
                    att_test, yh_test, test_rmse_epoch, test_mae_epoch, test_mape_epoch, yh_indi = reg.inference(xtest, \
                                                                                                                 ytest, \
                                                                                                                 [1.0, 1.0])
                    # [B V 1] - [B V] 
                    att_indi = np.squeeze(att_test, [2])
                
                    # dump file of truth, prediction and attentions 
                    y_yh = np.concatenate( [ytest, yh_test], 1 )
                    y_yh_att = np.concatenate( [y_yh, att_indi], 1 )
                    y_yh_att = np.concatenate( [y_yh_att, yh_indi], 1 )
            
                    y_yh_att.dump(pre_file + str(epoch) + ".dat")
                
                else:
                    att_test, yh_test, test_rmse_epoch, test_mae_epoch, test_mape_epoch = reg.inference(xtest, \
                                                                                                        ytest, \
                                                                                                        [1.0, 1.0])
                    #[B V 1] - [B V] 
                    att_indi = np.squeeze(att_test, [2])
                    
                    '''
                    # dump file of truth, prediction and attentions 
                    y_yh = np.concatenate( [ytest, yh_test], 1 )
                    y_yh_att = np.concatenate( [y_yh, att_indi], 1 )
            
                    y_yh_att.dump(pre_file + str(epoch) + ".dat")
                    '''
                
            train_rmse_epoch = sqrt(1.0*err_sum_epoch/total_cnt)
                
                
            # monitor training indicators
            print "\n --- At epoch %d: loss %f, train %s, test %s, %f, %f " % (epoch, loss_epoch*1.0/total_iter,\
                                                                        train_rmse_epoch, test_rmse_epoch, \
                                                                        test_mae_epoch,   test_mape_epoch ) 
            
            if method_str == 'clstm':
                print "     clstm regularized weights: ", regularized_weight
            
            
            print "     testing samples: ", np.squeeze(yh_test)[:5], np.squeeze(ytest)[:5]
            
            if para_attention != '':
                print "     attention samples: \n", att_indi[:5], '\n'
            
            
            # ---- logs of epoch performance    
            
            # write attention weights to txt files
            #if para_attention != '' :
            #    with open(attention_file, "a") as text_file:
            #        text_file.write("-- At epoch %d: %s \n" % (epoch, \
            #                                               (reg.test_attention(xtest[:1000], ytest[:1000], para_keep_prob)))) 
            
            # # each epoch, write training and testing errors to txt files
            
            if method_str == 'clstm':
                
                tmpstr=''
                for i in regularized_weight:
                    tmpstr = tmpstr + str(i)+','

                with open(log_file, "a") as text_file:
                    text_file.write("At epoch %d: loss %f, train %f, test %f, %f, %f, %s \n" 
                                    % (epoch, loss_epoch*1.0/total_iter, \
                                       train_rmse_epoch, test_rmse_epoch,\
                                       test_mae_epoch,   test_mape_epoch,\
                                       tmpstr) )
            else:
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
            
            #with open( pre_file + str(epoch) + ".txt", "w" ) as text_file:
            #    text_file.close()
            
            #with open( pre_file + str(epoch) + ".txt", "w" ) as text_file:
            #    text_file.write( "%s\n" %( str(list(zip(np.squeeze(ytest), np.squeeze(yh_test)))) ) )
            
            # ---
        
        print "Optimization Finished!"

    
# ----- loop over dropout and l2 

for tmp_keep_prob in [1.0, 0.8, 0.5, 0.3]:
    for tmp_l2 in [0.0001, 0.001, 0.01]:
        
        log_file = "res/" + str(dataset_str) + "_" + str(tmp_keep_prob) + str(tmp_l2) 
            
        pre_file = "res/pred/" + str(dataset_str) + "_" + str(tmp_keep_prob) + str(tmp_l2) + "_"
            
        train_nn( tmp_l2, tmp_keep_prob, log_file, pre_file ) 
        
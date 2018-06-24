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

    
# -- mv --

hidden_dim_dic = {}
hidden_dim_dic.update( {"energy": [210]} )
hidden_dim_dic.update( {"plant": [160]} )
hidden_dim_dic.update( {"pm25": [140]} )
hidden_dim_dic.update( {"syn": [220]} ) 
hidden_dim_dic.update( {"temp": [255]} ) 

# learning rate increases as network size 
lr_dic = {}
lr_dic.update( {"energy": 0.002} )
lr_dic.update( {"plant": 0.002} )
lr_dic.update( {"pm25": 0.002} )
lr_dic.update( {"syn": 0.001} ) 
lr_dic.update( {"temp": 0.001} ) 

# max_norm regularization
maxnorm_dic = {}
maxnorm_dic.update( {"energy": 3.0} )
maxnorm_dic.update( {"plant": 3.0} )
maxnorm_dic.update( {"pm25": 4.0} )
maxnorm_dic.update( {"syn": 4.0} )
maxnorm_dic.update( {"temp": 3.0} )


# attention type
attention_dic = {}
attention_dic.update( {"plain": "temp"} )
attention_dic.update( {"mv": "both-att"} )
attention_dic.update( {"clstm": ""} )
attention_dic.update( {"sep": "both-att"} )

# norm regularization
para_dense_regul_type_mv= 'l2'
# l1, l2
para_l2_att_mv = 0.00001

# layer normalization
para_layer_norm = ''

para_lstm_dims_mv = hidden_dim_dic[dataset_str] 
# energy 14 : 
# plant 8 : 
# pm25 7 : 
# synthetic :
# temp 17

# for both-fusion attention type 
para_dense_dims_mv = [ ]

para_lr_mv = lr_dic[dataset_str]
para_batch_size_mv = 64
    
para_attention_mv = attention_dic[method_str]
# temp, var, var-pre, both-att, both-pool, vari-mv-output'
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


# -- clstm --
    
para_attention_clstm = ''
para_lstm_dims_clstm = hidden_dim_dic[dataset_str]
para_dense_dims_clstm = [128, 64, 8]

para_lr_clstm = lr_dic[dataset_str]
para_batch_size_clstm = 64


# -- plain --
    
para_attention_plain = 'temp'
para_lstm_dims_plain = hidden_dim_dic[dataset_str]
para_dense_dims_plain = [ ]

para_lr_plain = lr_dic[dataset_str]
para_batch_size_plain = 64
    
# ------------------------------------------------

def train_nn( num_dense, l2_dense, dropout_keep_prob, log_file, pre_file ):   
    
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
                para_max_norm  = 0.0
            
            elif dropout_keep_prob == 0.8:
                
                para_keep_prob = [dropout_keep_prob, 1.0, 1.0]
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
                               para_attention_plain, para_l2_att_mv, num_dense)
            
            log_file += "_plain.txt"
            #model_file += "_plain"
            #attention_file += "_plain.txt"
            #PIK += "plain.dat"
            
            para_batch_size = para_batch_size_plain
            para_attention = para_attention_plain
            
            para_lstm_size = para_lstm_dims_plain
            para_lr = para_lr_plain
            
            
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
                                  para_bool_residual, attention_dic[method_str],\
                                  para_temp_attention_type, para_vari_attention_type,\
                                  para_dense_regul_type_mv, para_l2_att_mv, num_dense)
            
            log_file += "_sep.txt"
            #model_file += "_sep"
            #attention_file += "_sep.txt"
            #PIK += "sep.dat"
            
            para_batch_size = para_batch_size_sep
            para_attention = para_attention_plain
            para_lstm_size = para_lstm_dims_sep
            para_lr = para_lr_sep
        
        
        elif method_str == 'mv':
            
            # ?
            if dropout_keep_prob == 1.0:
                
                para_keep_prob = [1.0, 1.0, 1.0]
                para_max_norm = 0.0
            
            elif dropout_keep_prob == 0.8:
                
                para_keep_prob = [0.8, 0.8, 1.0]
                para_max_norm = maxnorm_dic[dataset_str]
                
            else:
                
                para_keep_prob = [dropout_keep_prob, dropout_keep_prob + 0.3, 1.0]
                para_max_norm = maxnorm_dic[dataset_str]
            
            print ' ---', method_str, ' parameter: ', \
            ' num of dense-', num_dense, \
            ' l2-', l2_dense, \
            ' dropout-', para_keep_prob, \
            ' maxnorm-', para_max_norm
            
            
            # --- learning rate decreases as number of layers is more
            lr_mv = 1.0*para_lr_mv/(num_dense+1.0)
            
            reg = tsLSTM_mv(para_dense_dims_mv, para_lstm_dims_mv, para_win_size, para_input_dim, sess, \
                            lr_mv, para_max_norm, para_batch_size_mv, para_bool_residual,\
                            para_attention_mv, para_temp_decay_type, para_temp_attention_type,\
                            l2_dense, para_l2_att_mv, para_vari_attention_type,\
                            para_loss_type, para_dense_regul_type_mv,\
                            para_layer_norm, num_dense )
            
            log_file += "_mv.txt"
            #model_file += "_mv"
            #attention_file += "_mv.txt"
            #PIK += "mv.dat"
            
            para_batch_size = para_batch_size_mv
            para_attention = para_attention_mv
            para_lstm_size = para_lstm_dims_mv
            para_lr = para_lr_mv
            
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
            
            para_lstm_size = para_lstm_dims_clstm
            para_lr = para_lr_clstm
            
            
        # parepare logs
        with open(log_file, "w") as text_file:
            text_file.close()
        
    
        # initialize the network
        reg.train_ini()
        reg.inference_ini()
        
        # perpare for data shuffling
        total_cnt  = np.shape(xtrain)[0]
        total_iter = int(total_cnt/para_batch_size)
        total_idx = range(total_cnt)
        
        # set up model saver
        saver = tf.train.Saver(max_to_keep = para_n_epoch)
        
        # epoch training and validation errors
        tr_rmse_epoch = []
        val_rmse_epoch = []
        val_mae_epoch = []
        
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
                
                if method_str == 'mv' and para_attention == 'both-att':
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
            
            
            tr_rmse_epoch.append([epoch, train_rmse_epoch])
            val_rmse_epoch.append([epoch, test_rmse_epoch])
            val_mae_epoch.append([epoch, test_mae_epoch])
            
                
            # monitor training indicators
            print "\n --- At epoch %d: loss %f, train %s, test %s, %f, %f " % (epoch, loss_epoch*1.0/total_iter,\
                                                                        train_rmse_epoch, test_rmse_epoch, \
                                                                        test_mae_epoch,   test_mape_epoch ) 
            if method_str == 'clstm':
                print "     clstm regularized weights: ", regularized_weight
            
            print "     testing samples: ", np.squeeze(yh_test)[:5], np.squeeze(ytest)[:5]
            
            if para_attention != '':
                print "     attention samples: \n     ", att_indi[:5], '\n'
                
                if (method_str == 'mv' and para_attention == 'both-att'):
                    print "     individual prediction:\n     ", yh_indi[:5], '\n'
            
            
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
                
        
        print "Optimization Finished!"
        
        return min(tr_rmse_epoch, key = lambda x:x[1]), min(val_rmse_epoch, key = lambda x:x[1]), \
               min(val_mae_epoch, key = lambda x:x[1])
                   
    
# ----- training loop

# prepare the log
with open("../../ts_results/ts_rnn.txt", "a") as text_file:
    text_file.write("\n%s %s  %s \n"%(dataset_str, method_str, attention_dic[method_str]))
    text_file.write("size: %s, lr: %s \n"%(str(hidden_dim_dic[dataset_str]), str(lr_dic[dataset_str])))
    text_file.write("data shape : %s \n"%(str(np.shape(xtrain))) )
    
log_rmse = []
log_mae = []

for tmp_num_dense in [0, 1, 2]:
    
    for tmp_keep_prob in [1.0, 0.8, 0.5]:
        
        for tmp_l2 in [0.0001, 0.001, 0.01]:
            
            log_file = "res/" + str(dataset_str) + "_" + str(tmp_num_dense) + str(tmp_keep_prob) + str(tmp_l2) 
            
            pre_file = "res/pred/" + str(dataset_str) + "_" + str(tmp_num_dense) + str(tmp_keep_prob) + str(tmp_l2) + "_"
            
            tmp_epoch_tr_rmse, tmp_epoch_rmse, tmp_epoch_mae = train_nn( tmp_num_dense, tmp_l2, tmp_keep_prob, log_file, pre_file ) 
            
            
            log_rmse.append( [tmp_num_dense, tmp_keep_prob, tmp_l2, tmp_epoch_rmse[0], tmp_epoch_rmse[1]] )
            log_mae.append( [tmp_num_dense, tmp_keep_prob, tmp_l2, tmp_epoch_mae[0], tmp_epoch_mae[1]] )
            
            
            print '\n--- current running: ', [tmp_num_dense, tmp_keep_prob, tmp_l2, tmp_epoch_rmse[0], tmp_epoch_rmse[1]], \
            [tmp_num_dense, tmp_keep_prob, tmp_l2, tmp_epoch_mae[0], tmp_epoch_mae[1]]
            
            with open("../../ts_results/ts_rnn.txt", "a") as text_file:
                text_file.write( "%f %f %f %s %s %s \n"%(tmp_num_dense, tmp_keep_prob, tmp_l2, \
                                                        str(tmp_epoch_tr_rmse), str(tmp_epoch_rmse), str(tmp_epoch_mae)) )  

with open("../../ts_results/ts_rnn.txt", "a") as text_file:
    text_file.write( "\n  ") 
            
            

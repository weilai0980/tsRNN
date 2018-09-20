#!/usr/bin/python
import sys
import collections
import hashlib
import numbers
import pickle

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.rnn_cell_impl import *

import time
import json

# local 
from mv_rnn_cell import *
from ts_mv_rnn import *
from ts_clstm import *
from utils_libs import *

# fix the random seed to reproduce the results
np.random.seed(1)
tf.set_random_seed(1)


# ------ Load data ------
    
dataset_str = str(sys.argv[1])
method_str = str(sys.argv[2])

with open('config.json') as f:
    file_dict = json.load(f)
    
print("--- Loading files at", file_dict[dataset_str]) 
files_list = file_dict[dataset_str]
    
xtrain = np.load(files_list[0], encoding='latin1')
xtest = np.load(files_list[1], encoding='latin1')
ytrain = np.load(files_list[2], encoding='latin1')
ytest = np.load(files_list[3], encoding='latin1')
    
# align the dimension    
ytrain = np.expand_dims(ytrain, 1) 
ytest  = np.expand_dims(ytest,  1)

# fixed
para_input_dim = np.shape(xtrain)[-1]
para_win_size = np.shape(xtrain)[1]

print("--- Data shapes: ", np.shape(xtrain), np.shape(ytrain), np.shape(xtest), np.shape(ytest))

    
# ------ network set-up ------

para_is_stateful = False
para_n_epoch = 80
    
# if residual layers are used, keep all dimensions the same 
para_bool_residual = False


# -- mv --

# 14 8 7 11 17 82 
hidden_dim_dic = {}

hidden_dim_dic.update( {"plant": [200]} )
hidden_dim_dic.update( {"plant_causal": [120]} )

hidden_dim_dic.update( {"pm25": [140]} )

hidden_dim_dic.update( {"nasdaq": [820]} )
hidden_dim_dic.update( {"sml": [170]} )

# learning rate increases as network size 
lr_dic = {}
lr_dic.update( {"plant": 0.003} )
lr_dic.update( {"plant_causal": 0.003} )

lr_dic.update( {"pm25": 0.002} )

lr_dic.update( {"nasdaq": 0.005} )
lr_dic.update( {"sml": 0.01} ) 

# max_norm regularization
maxnorm_dic = {}
maxnorm_dic.update( {"plant": 5.0} )
maxnorm_dic.update( {"plant_causal": 5.0} )

maxnorm_dic.update( {"pm25": 4.0} )
maxnorm_dic.update( {"nasdaq": 4.0} )
maxnorm_dic.update( {"sml": 5.0} )


# batch size 
batch_size_dic = {}
batch_size_dic.update( {"plant": 64} )
batch_size_dic.update( {"plant_causal": 64} )

batch_size_dic.update( {"pm25": 4.0} )
batch_size_dic.update( {"nasdaq": 4.0} )
batch_size_dic.update( {"sml": 32} )


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

# for both-fusion attention type 
# para_dense_dims_mv = [ ]

para_lr_mv = lr_dic[dataset_str]
para_batch_size_mv = batch_size_dic[dataset_str]

    
para_attention_mv = attention_dic[method_str]
# temp, var, var-pre, both-att, both-pool, vari-mv-output'
para_temp_attention_type = 'temp_loc'
# loc, concate
para_temp_decay_type = ''
# cutoff
para_vari_attention_type = 'vari_loc_all'
para_rnn_gate_type = 'tensor'
# full, tensor
para_pool_type = ''
para_loss_type = 'mse'
# mse,
# lk: likelihood 
para_loss_granularity = ''
# step, last

para_ke_type = 'aggre_posterior'
# base_posterior, base_prior

# -- seperate --

# per variable
para_lstm_dims_sep = [i/para_input_dim for i in hidden_dim_dic[dataset_str]]
#para_dense_dims_sep = [ ]

para_lr_sep = lr_dic[dataset_str]
para_batch_size_sep = 64


# -- clstm --
    
para_lstm_dims_clstm = hidden_dim_dic[dataset_str]
para_dense_dims_clstm = [128, 64, 8]

para_lr_clstm = lr_dic[dataset_str]
para_batch_size_clstm = 64


# -- plain --
    
para_attention_plain = 'temp'
para_lstm_dims_plain = hidden_dim_dic[dataset_str]

para_lr_plain = lr_dic[dataset_str]
para_batch_size_plain = 64
    
# ------------------------------------------------

def train_nn( num_dense, l2_dense, dropout_keep_prob, log_file, pre_file, ke_file, att_file, log_le ):   
    
    # ---- build and train the model ----
    
    # clear graph
    tf.reset_default_graph()
    
    # fix the random seed to stabilize the network 
    np.random.seed(1)
    tf.set_random_seed(1)
    
    with tf.device('/device:GPU:7'):
        
        config = tf.ConfigProto(allow_soft_placement = True)
        #config = config
        sess = tf.Session(config = config)
                
        # fix the random seed to stabilize the network 
        # np.random.seed(1)
        # tf.set_random_seed(1)
        
        if method_str == 'plain':
            
            # apply max_norm contraint only when dropout is used
            if dropout_keep_prob == 1.0:
                
                para_keep_prob = [1.0, 1.0, 1.0]
                para_max_norm  = 0.0
            
            elif dropout_keep_prob == 0.8:
                
                para_keep_prob = [dropout_keep_prob, 1.0, 1.0]
                para_max_norm = maxnorm_dic[dataset_str]
            
            else:
                
                para_keep_prob = [dropout_keep_prob, dropout_keep_prob + 0.3, 1.0]
                para_max_norm = maxnorm_dic[dataset_str]
                
            
            print(' ---', method_str, ' parameter: ', ' l2-', l2_dense, ' dropout-', para_keep_prob, 'maxnorm-', para_max_norm)
            
            reg = tsLSTM_plain(para_lstm_dims_plain, \
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
            
            # apply max_norm contraint only when dropout is used
            if dropout_keep_prob == 1.0:
                
                para_keep_prob = [1.0, 1.0, 1.0]
                para_max_norm = 0.0
            
            elif dropout_keep_prob == 0.8:
                
                para_keep_prob = [0.8, 0.8, 1.0]
                para_max_norm = maxnorm_dic[dataset_str]
                
            else:
                
                para_keep_prob = [dropout_keep_prob, dropout_keep_prob + 0.3, 1.0]
                para_max_norm = maxnorm_dic[dataset_str]
            
            
            print(' ---', method_str, ' parameter: ', ' l2-', l2_dense, ' dropout-', para_keep_prob, \
            ' maxnorm-', para_max_norm)
            
            reg = tsLSTM_seperate(para_lstm_dims_sep, \
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
            
            # apply max_norm contraint only when dropout is used
            if dropout_keep_prob == 1.0:
                
                para_keep_prob = [1.0, 1.0, 1.0]
                para_max_norm = 0.0
            
            elif dropout_keep_prob == 0.8:
                
                para_keep_prob = [0.8, 0.8, 1.0]
                para_max_norm = maxnorm_dic[dataset_str]
                
            else:
                
                para_keep_prob = [dropout_keep_prob, dropout_keep_prob + 0.3, 1.0]
                para_max_norm = maxnorm_dic[dataset_str]
            
            print('--- ', method_str, ' parameter: ', \
                  ' num of dense-', num_dense, \
                  ' l2-', l2_dense, \
                  ' dropout-', para_keep_prob, \
                  ' maxnorm-', para_max_norm)
            
            # ---- learning rate decreases as number of layers is more
            
            #lr_mv = 1.0*para_lr_mv/(num_dense+1.0)
            
            reg = tsLSTM_mv(para_lstm_dims_mv, para_win_size, para_input_dim, sess, \
                            para_lr_mv, para_max_norm, para_bool_residual,\
                            para_attention_mv, para_temp_decay_type, para_temp_attention_type,\
                            l2_dense, para_l2_att_mv, para_vari_attention_type,\
                            para_loss_type, para_dense_regul_type_mv,\
                            para_layer_norm, num_dense, para_ke_type, para_rnn_gate_type)
            
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
                
            print(' ---', method_str, ' parameter: ', ' l2-', l2_dense, ' dropout-', para_keep_prob, \
                  ' maxnorm-', para_max_norm)
            
            reg = cLSTM_causal(para_dense_dims_clstm, para_lstm_dims_clstm, \
                               para_win_size, para_input_dim, sess, \
                               para_lr_clstm, l2_dense, para_max_norm, \
                               para_batch_size_clstm, para_bool_residual, \
                               para_l2_att_mv, l2_dense)
            
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
        total_idx = list(range(total_cnt))
        
        # set up model saver
        saver = tf.train.Saver(max_to_keep = para_n_epoch)
        
        # epoch training and validation errors
        epoch_error = []
        epoch_ke = []
        epoch_att = []
        
        st_time = time.time()
        
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
                
                # training
                tmp_loss, tmp_err = reg.train_batch(batch_x, batch_y, para_keep_prob)
                
                loss_epoch += tmp_loss
                err_sum_epoch += tmp_err
            
            # epoch: learning rate update
            # if epoch >= 20:
            #    reg.train_update_optimizer(para_lr_mv)
            
            
            # ---- training data: variable wise temporal attention, varable attention, variable correlation positive/negative
            
            # true, prediction, attention 
            if para_attention == '':
                
                # [self.y_hat, self.rmse, self.mae, self.mape]
                yh_test, test_rmse_epoch, test_mae_epoch, test_mape_epoch = reg.inference(xtest, \
                                                                                          ytest, \
                                                                                          [1.0, 1.0])
            else:
                
                if method_str == 'mv' and para_attention == 'both-att':
                    
                    # [self.y_hat, self.rmse, self.mae, self.mape]
                    # [B V T-1], [B V]
                    yh_test, test_rmse_epoch, test_mae_epoch, test_mape_epoch = reg.inference(xtest, ytest, [1.0, 1.0])
                    
                    # knowledge extraction
                    test_w, att_temp, att_vari, importance_vari_temp, importance_vari_prior, importance_vari_posterior = \
                    reg.knowledge_extraction(xtrain, ytrain, [1.0, 1.0])
                    
                    epoch_att.append([att_temp, att_vari])
                    epoch_ke.append([importance_vari_temp, importance_vari_prior, importance_vari_posterior])
                    
                    with open(log_ke, "a") as text_file:
                        text_file.write("\n epoch %d: \n\n %s \n %s \n %s \n"%(epoch, 
                                                                         str(importance_vari_temp),
                                                                         str(importance_vari_prior),
                                                                         str(importance_vari_posterior)))
                
                else:
                    att_test, yh_test, test_rmse_epoch, test_mae_epoch, test_mape_epoch = reg.inference(xtest, \
                                                                                                        ytest, \
                                                                                                        [1.0, 1.0])
            train_rmse_epoch = sqrt(1.0*err_sum_epoch/total_cnt)
            epoch_error.append([epoch, loss_epoch*1.0/total_iter, train_rmse_epoch, [test_rmse_epoch, test_mae_epoch,\
                                                                                     test_mape_epoch]])
            
            print("\n --- At epoch %d : \n     %s " %(epoch, str(epoch_error[-1])), test_w)
            
            
            # ---- terminal output
            
            # prediction samples
            # print("     testing samples: ", np.squeeze(ytest)[:5], np.squeeze(yh_test)[:5])
            
            #if method_str == 'clstm':
            #    print("     clstm regularized weights: ", regularized_weight)
            
            #if para_attention != '':
            #    print("     attention samples: \n     ", att_indi[:5], '\n')
                
            #    if (method_str == 'mv' and para_attention == 'both-att'):
            #        print("     individual prediction:\n     ", yh_indi[:5], '\n')
            
        
        ed_time = time.time()
        
        print("Optimization Finished!") 
        
        
        # ---- dump epoch results
        
        if method_str == 'mv': 
            
            pickle.dump(epoch_ke, open(ke_file + ".p", "wb"))
            pickle.dump(epoch_att, open(att_file + ".p", "wb"))
            
        
        return min(epoch_error, key = lambda x: x[3][0]), 1.0*(ed_time - st_time)/para_n_epoch
    
    '''
            # testing data true, prediction, attention 
            if para_attention == '':
                
                # [self.y_hat, self.rmse, self.mae, self.mape]
                yh_test, test_rmse_epoch, test_mae_epoch, test_mape_epoch = reg.inference(xtest, ytest, [1.0, 1.0])
                
                if method_str == 'mv': 
                    # dump file of truth and prediction
                    y_yh = np.concatenate( [ytest, yh_test], 1 )
                    y_yh.dump(pre_file + str(epoch) + ".dat")
                    
                    
                    #att_file
                    
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
            
            
            # ---- logs of epoch performance    
            
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
            
            elif method_str == 'mv' and para_attention == 'both-att':
                
                # summerized temporal attention, variable attention
                
                with open(log_file, "a") as text_file:
                    text_file.write("At epoch %d: loss %f, train %f, test %f, %f, %f \n" % (epoch, loss_epoch*1.0/total_iter, \
                                                                               train_rmse_epoch, test_rmse_epoch,\
                                                                               test_mae_epoch,   test_mape_epoch) )
    '''
                   
# ----- training loop

if __name__ == '__main__':
    
    # log: hyperparameter
    with open("../../ts_results/ts_rnn.txt", "a") as text_file:
        text_file.write("\n-- dataset name: %s \n"%(dataset_str))
        text_file.write("dataset shape: %s \n"%(str(np.shape(xtrain))))
        text_file.write("%s %s  \n"%(method_str, attention_dic[method_str]))
        text_file.write("size: %s, lr: %s \n"%(str(hidden_dim_dic[dataset_str]), str(lr_dic[dataset_str])))
        text_file.write("attention: %s, %s \n"%(para_temp_attention_type, para_vari_attention_type))
        text_file.write("loss type: %s \n"%(para_loss_type))
        text_file.write("batch size: %s \n"%(str(para_batch_size_mv)))
        text_file.write("knowledge extraction type : %s \n"%(para_ke_type))
        text_file.write("rnn gate type : %s \n"%(para_rnn_gate_type))
        text_file.write("maximum norm constraint : %f \n"%(maxnorm_dic[dataset_str]))
        
        
    # grid search process
    validate_tuple = []
    
    log_ke = "../../ts_results/log_ke_" + str(dataset_str) + ".txt"
    with open(log_ke, "w") as text_file:
        text_file.write("")
    
    #for para_lr_mv in [0.001, 0.002, 0.005]
    for tmp_num_dense in [0, 1]:
        for tmp_keep_prob in [1.0, 0.8, 0.5]:
            for tmp_l2 in [0.0001, 0.001, 0.01, 0.1]:
                
                #para_lr_mv = para_lr_mv * (tmp_num_dense + 1)
                
                # -- epoch log files
                
                log_epoch_file = "../../ts_results/log_"\
                                 + str(dataset_str) + "_" + str(tmp_num_dense) + str(tmp_keep_prob) + str(tmp_l2)
                    
                pre_file = "../../ts_results/pred_"\
                           + str(dataset_str) + "_" + str(tmp_num_dense) + str(tmp_keep_prob) + str(tmp_l2) + "_"
                
                # data file: ke - knowledge extraction
                ke_data = "../../ts_results/ke_" \
                + str(dataset_str) + "_" + str(tmp_num_dense) + str(tmp_keep_prob) + str(tmp_l2) + "_"
                
                # data file: att - temporal and variable attention
                att_data = "../../ts_results/att_" \
                + str(dataset_str) + "_" + str(tmp_num_dense) + str(tmp_keep_prob) + str(tmp_l2) + "_"
                
                with open(log_ke, "a") as text_file:
                    text_file.write("\n -------- %f %f %f \n\n"%(tmp_num_dense, tmp_keep_prob, tmp_l2))    
                
                
                # --- training 
                error_tuple, epoch_time = train_nn(tmp_num_dense, tmp_l2, tmp_keep_prob, log_epoch_file, pre_file, \
                                                   ke_data, att_data, log_ke)
                
                validate_tuple.append(error_tuple) 
            
                
                print('\n --- current running: ', tmp_num_dense, tmp_keep_prob, tmp_l2, validate_tuple[-1], '\n')
                
                # log: performance for one hyperparameter set-up
                with open("../../ts_results/ts_rnn.txt", "a") as text_file:
                    text_file.write( "%f %f %f %s %s \n"%(tmp_num_dense, tmp_keep_prob, tmp_l2, 
                                                          str(validate_tuple[-1]), str(epoch_time)) )
                    
    with open("../../ts_results/ts_rnn.txt", "a") as text_file:
        text_file.write( "\n  ") 

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

''' 
Arguments:

dataset_str: name of the dataset
method_str: name of the neural network 

'''

# ------ load data ------

# parameters from command line
dataset_str = str(sys.argv[1])
method_str = str(sys.argv[2])

# parameters from config.json
import json
with open('config_data.json') as f:
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

print(" --- Data shapes: ", np.shape(xtrain), np.shape(ytrain), np.shape(xtest), np.shape(ytest))
    
# ------ network set-up ------

para_is_stateful = False
para_n_epoch = 100
    
# if residual layers are used, keep all dimensions the same 
para_bool_residual = False

# -- mv --

# size of recurrent layers    
hidden_dim_dic = {}

hidden_dim_dic.update( {"plant": [200]} )
hidden_dim_dic.update( {"plant_pearson": [120]} )
hidden_dim_dic.update( {"plant_no_target": [180]} )
hidden_dim_dic.update( {"plant_sub_full": [120]} )
hidden_dim_dic.update( {"plant_sub_tensor": [120]} )
hidden_dim_dic.update( {"plant_uni": [128]} )

hidden_dim_dic.update( {"nasdaq": [820]} )
hidden_dim_dic.update( {"nasdaq_pearson": [410]} )
hidden_dim_dic.update( {"nasdaq_no_target": [810]} )
hidden_dim_dic.update( {"nasdaq_sub_full": [410]} )
hidden_dim_dic.update( {"nasdaq_sub_tensor": [410]} )
hidden_dim_dic.update( {"nasdaq_uni": [128]} )

hidden_dim_dic.update( {"sml": [170]} )
hidden_dim_dic.update( {"sml_pearson": [90]} )
hidden_dim_dic.update( {"sml_no_target": [160]} )
hidden_dim_dic.update( {"sml_sub_full": [90]} )
hidden_dim_dic.update( {"sml_sub_tensor": [90]} )
hidden_dim_dic.update( {"sml_uni": [128]} )


# learning rate increases as network size 
lr_dic = {}

lr_dic.update( {"plant": 0.5} )
lr_dic.update( {"plant_pearson": 0.005} )
lr_dic.update( {"plant_no_target": 0.005} )
lr_dic.update( {"plant_sub_full": 0.005} )
lr_dic.update( {"plant_sub_tensor": 0.005} )
lr_dic.update( {"plant_uni": 0.005} )

lr_dic.update( {"sml": 0.01} )
lr_dic.update( {"sml_pearson": 0.01} )
lr_dic.update( {"sml_no_target": 0.01} )
lr_dic.update( {"sml_sub_full": 0.01} )
lr_dic.update( {"sml_sub_tensor": 0.01} )
lr_dic.update( {"sml_uni": 0.01} )

lr_dic.update( {"nasdaq": 0.05} )
lr_dic.update( {"nasdaq_pearson": 0.05} )
lr_dic.update( {"nasdaq_no_target": 0.05} )
lr_dic.update( {"nasdaq_sub_full": 0.05} )
lr_dic.update( {"nasdaq_sub_tensor": 0.05} )
lr_dic.update( {"nasdaq_uni": 0.05} )

'''
lr_dic.update( {"pm25": 0.002} )
lr_dic.update( {"pm25": 0.002} )
lr_dic.update( {"pm25_sub_full": 0.002} )
lr_dic.update( {"pm25_sub_tensor": 0.002} )
'''

# batch size 
batch_size_dic = {}
batch_size_dic.update( {"plant": 64} )
batch_size_dic.update( {"plant_pearson": 64} )
batch_size_dic.update( {"plant_no_target": 64} )
batch_size_dic.update( {"plant_sub_full": 64} )
batch_size_dic.update( {"plant_sub_tensor": 64} )
batch_size_dic.update( {"plant_uni": 64} )

batch_size_dic.update( {"nasdaq": 64} )
batch_size_dic.update( {"nasdaq_pearson": 64} )
batch_size_dic.update( {"nasdaq_no_target": 64} )
batch_size_dic.update( {"nasdaq_sub_full": 64} )
batch_size_dic.update( {"nasdaq_sub_tensor": 64} )
batch_size_dic.update( {"nasdaq_uni": 64} )

batch_size_dic.update( {"sml": 32} )
batch_size_dic.update( {"sml_pearson": 32} )
batch_size_dic.update( {"sml_no_target": 32} )
batch_size_dic.update( {"sml_sub_full": 32} )
batch_size_dic.update( {"sml_sub_tensor": 32} )
batch_size_dic.update( {"sml_uni": 32} )

'''
batch_size_dic.update( {"pm25": 32} )
batch_size_dic.update( {"pm25": 32} )
batch_size_dic.update( {"pm25_no_target": 32} )
batch_size_dic.update( {"pm25_sub_full": 32} )
batch_size_dic.update( {"pm25_sub_tensor": 32} )
'''

# max_norm contraints
maxnorm_dic = {}

maxnorm_dic.update( {"plant": 5.0} )
maxnorm_dic.update( {"plant_pearson": 5.0} )
maxnorm_dic.update( {"plant_no_target": 5.0} )
maxnorm_dic.update( {"plant_sub_full": 5.0} )
maxnorm_dic.update( {"plant_sub_tensor": 5.0} )
maxnorm_dic.update( {"plant_uni": 5.0} )

maxnorm_dic.update( {"sml": 5.0} )
maxnorm_dic.update( {"sml_pearson": 5.0} )
maxnorm_dic.update( {"sml_no_target": 5.0} )
maxnorm_dic.update( {"sml_sub_full": 5.0} )
maxnorm_dic.update( {"sml_sub_tensor": 5.0} )
maxnorm_dic.update( {"sml_uni": 5.0} )

maxnorm_dic.update( {"nasdaq": 5.0} )
maxnorm_dic.update( {"nasdaq_pearson": 5.0} )
maxnorm_dic.update( {"nasdaq_no_target": 5.0} )
maxnorm_dic.update( {"nasdaq_sub_full": 5.0} )
maxnorm_dic.update( {"nasdaq_sub_tensor": 5.0} )
maxnorm_dic.update( {"nasdaq_uni": 5.0} )

'''
maxnorm_dic.update( {"pm25": 4.0} )
maxnorm_dic.update( {"pm25_sub_full": 4.0} )
maxnorm_dic.update( {"pm25_sub_tensor": 4.0} )
'''

# attention type
attention_dic = {}
attention_dic.update( {"plain": "temp"} )
attention_dic.update( {"mv_full": "both-att"} )
attention_dic.update( {"mv_tensor": "both-att"} )

'''
attention_dic.update( {"clstm": ""} )
attention_dic.update( {"clstm_sub": ""} )

attention_dic.update( {"sep": "both-att"} )
attention_dic.update( {"sep_sub": "both-att"} )
'''

# regularization
para_dense_regul_type_mv= 'l2'  # l1, l2
para_l2_att_mv = 0.00001

para_bool_regular_lstm = True
para_bool_regular_attention = False

# layer normalization
para_layer_norm = ''

# learning rate, convergence
para_lr_mv = lr_dic[dataset_str]
para_batch_size_mv = batch_size_dic[dataset_str]
para_decay_step = 700000

# multi-variable architecture
para_rnn_gate_type = "full" if method_str == 'mv_full' else 'tensor'

para_lstm_dims_mv = hidden_dim_dic[dataset_str] 

para_attention_mv = attention_dic[method_str] # temp, var, var-pre, both-att, both-pool, vari-mv-output'
para_temp_attention_type = 'temp_loc' # loc, concate
para_temp_decay_type = ''  # cutoff
para_vari_attention_type = 'vari_loc_all'
                  
para_loss_type = 'pseudo_lk' # mse, lk: likelihood, pseudo_lk 

para_ke_type = 'aggre_posterior' # base_posterior, base_prior

'''
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
'''

# -- plain --
    
para_attention_plain = attention_dic[method_str]
para_lstm_dims_plain = hidden_dim_dic[dataset_str]

para_lr_plain = lr_dic[dataset_str]
para_batch_size_plain = 64
    
# ------------------------------------------------

def train_nn(num_dense, l2_dense, dropout_keep_prob, log_file, ke_pickle, test_pickle):   
    
    # ---- build and train the model ----
    
    # clear graph
    tf.reset_default_graph()
    
    # fix the random seed to stabilize the network 
    np.random.seed(1)
    tf.set_random_seed(1)
    
    with tf.device('/device:GPU:5'):
        
        config = tf.ConfigProto(allow_soft_placement = True)
        #config = config
        
        sess = tf.Session(config = config)
        #sess = tf.Session()
                
        # fix the random seed to stabilize the network 
        # np.random.seed(1)
        # tf.set_random_seed(1)
        
        # apply max_norm contraint only when dropout is used
        para_keep_prob = [dropout_keep_prob, min(1.0, dropout_keep_prob + 0.2), 1.0]
        para_max_norm = maxnorm_dic[dataset_str] if dropout_keep_prob < 1.0 else 0.0
        
        if method_str == 'plain':
            
            print('\n\n ---', method_str, ' parameter: ',\
                  ' l2-', l2_dense, 
                  ' dropout-', para_keep_prob, 
                  'maxnorm-', para_max_norm)
            
            reg = tsLSTM_plain(para_lstm_dims_plain, \
                               para_win_size, para_input_dim, sess, \
                               para_lr_plain, l2_dense, para_max_norm, \
                               para_bool_residual, \
                               para_attention_plain, para_l2_att_mv, num_dense)
            
            para_batch_size = para_batch_size_plain
            para_attention = para_attention_plain
        
        elif method_str == 'mv_full':
            
            print('\n\n --- ', method_str, ' parameter: ',\
                  ' num of dense-', num_dense,\
                  ' l2-', l2_dense,\
                  ' dropout-', para_keep_prob,\
                  ' maxnorm-', para_max_norm)
            
            reg = tsLSTM_mv(para_lstm_dims_mv, para_win_size, para_input_dim, sess, \
                            para_lr_mv, para_max_norm, para_bool_residual,\
                            para_attention_mv, para_temp_decay_type, para_temp_attention_type,\
                            l2_dense, para_l2_att_mv, para_vari_attention_type,\
                            para_loss_type, para_dense_regul_type_mv,\
                            para_layer_norm, num_dense, para_ke_type, "full")
            
            para_batch_size = para_batch_size_mv
            para_attention = para_attention_mv
            
        elif method_str == 'mv_tensor':
            
            print('\n\n --- ', method_str, ' parameter: ', \
                  ' num of dense-', num_dense, \
                  ' l2-', l2_dense, \
                  ' dropout-', para_keep_prob, \
                  ' maxnorm-', para_max_norm)
            
            reg = tsLSTM_mv(para_lstm_dims_mv, para_win_size, para_input_dim, sess, \
                            para_lr_mv, para_max_norm, para_bool_residual,\
                            para_attention_mv, para_temp_decay_type, para_temp_attention_type,\
                            l2_dense, para_l2_att_mv, para_vari_attention_type,\
                            para_loss_type, para_dense_regul_type_mv,\
                            para_layer_norm, num_dense, para_ke_type, "tensor")
            
            para_batch_size = para_batch_size_mv
            para_attention = para_attention_mv
        
        '''    
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
            
        '''
        
        # initialize the network
        reg.train_ini()
        reg.inference_ini()
        
        # perpare for data shuffling
        total_cnt = np.shape(xtrain)[0]
        iter_per_epoch = int(total_cnt/para_batch_size) + 1
        total_idx = list(range(total_cnt))
        
        # set up model saver
        saver = tf.train.Saver(max_to_keep = para_n_epoch)
        
        # epoch training and validation errors
        epoch_error = []
        epoch_ke = []
        epoch_att = []
        epoch_test_prediction = []
        
        st_time = time.time()
        
        # training epoches 
        for epoch in range(para_n_epoch):
            
            st_time_epoch = time.time()
            
            loss_epoch = 0.0
            err_sum_epoch = 0.0
            
            # -- batch training
            
            np.random.shuffle(total_idx)
            
            for i in range(iter_per_epoch):
                
                # shuffle training data
                batch_idx = total_idx[ i*para_batch_size: min((i+1)*para_batch_size, total_cnt) ] 
                batch_x = xtrain[ batch_idx ]
                batch_y = ytrain[ batch_idx ]            
                
                if method_str == 'mv_full' or method_str == 'mv_tensor':
                    
                    # learning rate decay
                    if (i + iter_per_epoch*epoch) != 0 and (i + iter_per_epoch*epoch)%para_decay_step == 0:
                        tmp_loss, tmp_err = reg.train_batch(batch_x, \
                                                        batch_y, \
                                                        para_keep_prob, \
                                                        True, \
                                                        para_lr_mv*(0.96)**((i + iter_per_epoch*epoch)/para_decay_step))
                    else:
                        tmp_loss, tmp_err = reg.train_batch(batch_x, \
                                                        batch_y, \
                                                        para_keep_prob, \
                                                        False, \
                                                        0.0)
                else:
                    tmp_loss, tmp_err = reg.train_batch(batch_x, batch_y, para_keep_prob)
                
                loss_epoch += tmp_loss
                err_sum_epoch += tmp_err
                       
            # -- epoch-wise evaluation
            
            if method_str == 'mv_full' or method_str == 'mv_tensor':
                
                if para_attention == 'both-att':
                    
                    # [self.y_hat, self.rmse, self.mae, self.mape]
                    # [B V T-1], [B V]
                    yh_test, test_rmse_epoch, test_mae_epoch, test_mape_epoch = reg.inference(xtest, ytest, [1.0, 1.0])
                    
                    # knowledge extraction
                    test_w, att_temp, att_prior, att_poster, importance_vari_temp, importance_vari_prior,\
                    importance_vari_posterior = reg.knowledge_extraction(xtrain, ytrain, [1.0, 1.0])
                    
                    #epoch_att.append([att_temp, att_prior, att_poster])
                    epoch_ke.append([importance_vari_temp, importance_vari_prior, importance_vari_posterior])
            
            else:
                yh_test, test_rmse_epoch, test_mae_epoch, test_mape_epoch = reg.inference(xtest, \
                                                                                          ytest, \
                                                                                          [1.0, 1.0])
            ed_time_epoch = time.time()
            
            train_rmse_epoch = sqrt(1.0*err_sum_epoch/total_cnt)
            
            # test ?
            epoch_test_prediction.append(yh_test)
            epoch_error.append([epoch, \
                                loss_epoch*1.0/iter_per_epoch, \
                                train_rmse_epoch, \
                                test_rmse_epoch, \
                                test_mae_epoch, \
                                test_mape_epoch])
            # epoch-wise 
            print("\n --- At epoch %d : \n    %s, %d "%(epoch, str(epoch_error[-1][1:]), ed_time_epoch - st_time_epoch))
            
            with open(log_file, "a") as text_file:
                text_file.write("%s\n"%(str(epoch_error[-1])[1:-1]))
            
        ed_time = time.time()
        
        print("Optimization Finished!") 
        
        # ---- dump epoch results
        
        if (method_str == 'mv_full' or method_str == 'mv_tensor') and para_attention == 'both-att':            
            
            pickle.dump(epoch_ke, open(ke_pickle + ".p", "wb"))
            
            best_epoch = min(epoch_error, key = lambda x:x[3])[0]
            pickle.dump(list(zip(np.squeeze(ytest), np.squeeze(epoch_test_prediction[best_epoch]))), \
                        open(test_pickle + ".p", "wb"))
            
        return min(epoch_error, key = lambda x: x[3]), 1.0*(ed_time - st_time)/para_n_epoch
    
                   
# ---- training loop

'''
Log files:

ts_rnn.txt: overall errors, all method, all set-up

log_method_dataset: epoch level training errors, method dataset wise

ke_pickle: only for MV-RNN, set-up wise

test_pickle: only for MV-RNN, set-up wise

'''

if __name__ == '__main__':
    
    # log: overall erros, hyperparameter
    with open("../../ts_results/ts_rnn.txt", "a") as text_file:
        text_file.write("\n-- dataset: %s \n"%(dataset_str))
        text_file.write("dataset shape: %s \n"%(str(np.shape(xtrain))))
        text_file.write("%s %s  \n"%(method_str, attention_dic[method_str]))
        text_file.write("size: %s, lr: %s \n"%(str(hidden_dim_dic[dataset_str]), str(lr_dic[dataset_str])))
        text_file.write("attention: %s, %s \n"%(para_temp_attention_type, para_vari_attention_type))
        text_file.write("loss type: %s \n"%(para_loss_type))
        text_file.write("batch size: %s \n"%(str(para_batch_size_mv)))
        text_file.write("knowledge extraction type : %s \n"%(para_ke_type))
        text_file.write("rnn gate type : %s \n"%(para_rnn_gate_type))
        text_file.write("maximum norm constraint : %f \n"%(maxnorm_dic[dataset_str]))
        text_file.write("number of epoch : %f \n\n"%(para_n_epoch))
        
    # log: epoch files
    log_epoch_file = "../../ts_results/log_" + method_str + "_" + dataset_str + ".txt"
    with open(log_epoch_file, "a") as text_file:
        text_file.write("\n-- dataset: %s \n"%(dataset_str))
        text_file.write("dataset shape: %s \n"%(str(np.shape(xtrain))))
        text_file.write("%s %s  \n"%(method_str, attention_dic[method_str]))
        text_file.write("size: %s, lr: %s \n"%(str(hidden_dim_dic[dataset_str]), str(lr_dic[dataset_str])))
        text_file.write("attention: %s, %s \n"%(para_temp_attention_type, para_vari_attention_type))
        text_file.write("loss type: %s \n"%(para_loss_type))
        text_file.write("batch size: %s \n"%(str(para_batch_size_mv)))
        text_file.write("knowledge extraction type : %s \n"%(para_ke_type))
        text_file.write("rnn gate type : %s \n"%(para_rnn_gate_type))
        text_file.write("maximum norm constraint : %f \n"%(maxnorm_dic[dataset_str]))
        text_file.write("number of epoch : %f \n\n"%(para_n_epoch))

    # grid search process
    validate_tuple = []
    
    #for para_lr_mv in [0.001, 0.002, 0.005]
    for tmp_num_dense in [0]:
        for tmp_keep_prob in [1.0, 0.8]:
            for tmp_l2 in [0.00001, 0.0001, 0.001, 0.01]:
                
                with open(log_epoch_file, "a") as text_file:
                    text_file.write("\n num_dense: %d, keep_prob: %f, l2: %f \n"%(tmp_num_dense, tmp_keep_prob, tmp_l2))
                
                # pickle: ke - knowledge extraction
                ke_file = "../../ts_results/ke_" + \
                          str(para_rnn_gate_type) + "_" \
                          + str(dataset_str) + "_" \
                          + str(tmp_num_dense) + \
                            str(tmp_keep_prob) + \
                            str(tmp_l2) + "_"
                            
                # pickle: testing predictions            
                test_file = "../../ts_results/test_" + \
                          str(para_rnn_gate_type) + "_" \
                          + str(dataset_str) + "_" \
                          + str(tmp_num_dense) + \
                            str(tmp_keep_prob) + \
                            str(tmp_l2) + "_"
             
                # -- training
                
                error_tuple, epoch_time = train_nn(tmp_num_dense, 
                                                   tmp_l2, 
                                                   tmp_keep_prob, 
                                                   log_epoch_file,
                                                   ke_file,
                                                   test_file)
                
                validate_tuple.append(error_tuple) 
                
                print('\n --- current running: ', tmp_num_dense, tmp_keep_prob, tmp_l2, validate_tuple[-1], '\n')
                
                # log: overall errors, performance for one hyperparameter set-up
                with open("../../ts_results/ts_rnn.txt", "a") as text_file:
                    text_file.write( "%f %f %f %s %s \n"%(tmp_num_dense, 
                                                          tmp_keep_prob, 
                                                          tmp_l2, 
                                                          str(validate_tuple[-1]), 
                                                          str(epoch_time)))
                    
    with open("../../ts_results/ts_rnn.txt", "a") as text_file:
        text_file.write( "\n  ") 

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
from config_hyper_para import *

# fix the random seed to reproduce the results
np.random.seed(1)
tf.set_random_seed(1)

''' 
Arguments:

dataset_str: name of the dataset
method_str: name of the neural network 

'''

# ------ GPU set-up in multi-GPU environment ------

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5" 


# ------ load data ------

# parameters from command line
dataset_str = str(sys.argv[1])
method_str = str(sys.argv[2])

# parameters from config.json
with open('config_data.json') as f:
    file_dict = json.load(f)
    
print(" --- Loading files at", file_dict[dataset_str]) 

files_list = file_dict[dataset_str]    
xtrain = np.load(files_list[0], encoding='latin1')
xval = np.load(files_list[1], encoding='latin1')
ytrain = np.load(files_list[2], encoding='latin1')
yval = np.load(files_list[3], encoding='latin1')
    
# align the dimension    
ytrain = np.expand_dims(ytrain, 1) 
yval  = np.expand_dims(yval,  1)

# fixed
para_input_dim = np.shape(xtrain)[-1]
para_win_size = np.shape(xtrain)[1]

print(" --- Data shapes: ", np.shape(xtrain), np.shape(ytrain), np.shape(xval), np.shape(yval))

# ------ model set-up ------

# convergence
para_n_epoch = 100
para_decay_step = 1000000
para_lr_plain = lr_dic[dataset_str]
para_batch_size_plain = 64
para_is_stateful = False

# regularization
para_bool_regular_lstm = True
para_bool_regular_attention = False
para_bool_regular_dropout_output = False

    
# if residual layers are used, keep all dimensions the same 
para_bool_residual = False

# loss
para_loss_type = 'mse' # mse, lk: likelihood, pseudo_lk 

# attention
para_attention_plain = attention_dic[method_str]

# architecture 
para_lstm_dims_plain = hidden_dim_dic[dataset_str]


# ------ utility functions ------

def train_nn(num_dense, l2_dense, dropout_keep_prob, log_file, test_pickle):   
    
    # ---- build and train the model ----
    
    # clear graph
    tf.reset_default_graph()
    
    # fix the random seed to stabilize the network 
    np.random.seed(1)
    tf.set_random_seed(1)
    
    with tf.device('/device:GPU:0'):
        
        # session set-up
        config = tf.ConfigProto()
        
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        
        sess = tf.Session(config = config)
        
        # apply max_norm contraint only when dropout is used
        para_keep_prob = dropout_keep_prob
        para_max_norm = maxnorm_dic[dataset_str] if dropout_keep_prob < 1.0 else 0.0
        
        if method_str == 'plain':
            
            print('\n\n ---', method_str, ' parameter: ',\
                  ' l2-', l2_dense, 
                  ' dropout-', para_keep_prob, 
                  'maxnorm-', para_max_norm)
            
            reg = tsLSTM_plain(para_lstm_dims_plain, \
                               para_win_size, 
                               para_input_dim, 
                               sess, \
                               para_lr_plain, 
                               l2_dense, 
                               para_max_norm, \
                               para_bool_residual, \
                               para_attention_plain, 
                               para_l2_att_mv, 
                               num_dense,
                               para_bool_regular_attention, 
                               para_bool_regular_lstm, 
                               para_bool_regular_dropout_output)
            
            para_batch_size = para_batch_size_plain
        
        # initialize training and inference
        reg.train_ini()
        reg.inference_ini()
        
        # data shuffling parameters
        total_cnt = np.shape(xtrain)[0]
        iter_per_epoch = int(total_cnt/para_batch_size) + 1
        total_idx = list(range(total_cnt))
        
        # model saver
        saver = tf.train.Saver(max_to_keep = para_n_epoch)
        
        # epoch training and validation errors
        epoch_error = []
        epoch_test_prediction = []
        
        st_time = time.time()
        
        # training epoches 
        for epoch in range(para_n_epoch):
            
            st_time_epoch = time.time()
            
            loss_epoch = 0.0
            err_sum_epoch = 0.0
            
            # -- batch training
            
            # re-shuffle training data
            np.random.shuffle(total_idx)
            
            for i in range(iter_per_epoch):
                
                # batch data
                batch_idx = total_idx[ i*para_batch_size: min((i+1)*para_batch_size, total_cnt) ] 
                batch_x = xtrain[ batch_idx ]
                batch_y = ytrain[ batch_idx ]
                
                tmp_loss, tmp_err = reg.train_batch(batch_x, batch_y, para_keep_prob)
                
                loss_epoch += tmp_loss
                err_sum_epoch += tmp_err
                       
            # -- epoch-wise evaluation
            yh_val, val_rmse_epoch, val_mae_epoch, val_mape_epoch = reg.inference(xval, yval, [1.0, 1.0])
            ed_time_epoch = time.time()
            
            train_rmse_epoch = sqrt(1.0*err_sum_epoch/total_cnt)
            
            # test ?
            epoch_test_prediction.append(yh_val)
            epoch_error.append([epoch, \
                                loss_epoch*1.0/iter_per_epoch, \
                                train_rmse_epoch, \
                                val_rmse_epoch, \
                                val_mae_epoch, \
                                val_mape_epoch])
            # epoch-wise 
            print("\n --- At epoch %d : \n    %s, %d "%(epoch, str(epoch_error[-1][1:]), ed_time_epoch - st_time_epoch))
            
            with open(log_file, "a") as text_file:
                text_file.write("%s\n"%(str(epoch_error[-1])[1:-1]))
            
        ed_time = time.time()
        
        print("Optimization Finished!") 
        
        # ---- return results
        
        return min(epoch_error, key = lambda x: x[3]), 1.0*(ed_time - st_time)/para_n_epoch
    

def log_func(text_env):    
    
    text_env.write("\n---- dataset: %s \n"%(dataset_str))
    text_env.write("dataset shape: %s \n"%(str(np.shape(xtrain))))
    text_env.write("method: %s, %s  \n"%(method_str, attention_dic[method_str]))
    text_env.write("plain layer size: %s \n"%(str(hidden_dim_dic[dataset_str])))
    text_env.write("lr: %s \n"%(str(lr_dic[dataset_str])))
    text_env.write("learnign rate decay iterations : %d \n"%(para_decay_step))
    text_env.write("attention: %s, %s \n"%(para_temp_attention_type, para_vari_attention_type))
    text_env.write("loss type: %s \n"%(para_loss_type))
    text_env.write("batch size: %s \n"%(str(para_batch_size_mv)))
    
    text_env.write("maximum norm constraint : %f \n"%(maxnorm_dic[dataset_str]))
    text_env.write("number of epoch : %d \n"%(para_n_epoch))
    text_env.write("regularization on LSTM weights : %s \n"%(para_bool_regular_lstm))
    text_env.write("regularization on attention : %s \n"%(para_bool_regular_attention))
    text_env.write("dropout before the outpout layer : %s \n"%(para_bool_regular_dropout_output))
    
    return

# ---- training loop

'''
Log and dump files:

ts_rnn.txt: overall errors, all method, all set-up

log_method_dataset: epoch level training errors, method dataset wise

pred_pickle: only for MV-RNN, set-up wise

'''


if __name__ == '__main__':
    
    # log: overall erros, hyperparameter
    log_err_file = "../../ts_results/ts_rnn.txt"
    
    with open(log_err_file, "a") as text_file:
        log_func(text_file)
        
    # log: epoch files
    log_epoch_file = "../../ts_results/log_" + method_str + "_" + dataset_str + ".txt"
    
    with open(log_epoch_file, "a") as text_file:
        log_func(text_file)
        
    
    # grid search process
    validate_tuple = []
    
    #for para_lr_mv in [0.001, 0.002, 0.005]
    for tmp_num_dense in [0, 1]:
        for tmp_keep_prob in [1.0, 0.8, 0.5]:
            for tmp_l2 in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
                
                # log: epoch errors
                with open(log_epoch_file, "a") as text_file:
                    text_file.write("\n num_dense: %d, keep_prob: %f, l2: %f \n"%(tmp_num_dense, tmp_keep_prob, tmp_l2))
                
                            
                # pickle: predictions            
                pred_pickle = "../../ts_results/pred_" + \
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
                                                   pred_pickle)
                
                validate_tuple.append(error_tuple) 
                
                print('\n --- current running: ', tmp_num_dense, tmp_keep_prob, tmp_l2, validate_tuple[-1], '\n')
                
                # log: overall errors, hyperparameter set-up
                with open(log_err_file, "a") as text_file:
                    text_file.write( "%f %f %f %s %s \n"%(tmp_num_dense, 
                                                          tmp_keep_prob, 
                                                          tmp_l2, 
                                                          str(validate_tuple[-1]), 
                                                          str(epoch_time)))
                    
    with open("../../ts_results/ts_rnn.txt", "a") as text_file:
        text_file.write( "\n  ") 

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
from ts_mv_rnn import *
from ts_clstm import *
from utils_libs import *
from ts_mv_rnn_testing import *

from config_hyper_para_plain import *

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
os.environ["CUDA_VISIBLE_DEVICES"] = "4" 

# ------ arguments ------

# from command line
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', help = "dataset", type = str)
parser.add_argument('--model', '-m', help = "model", type = str, default = 'plain')
 
args = parser.parse_args()
#print(parser.format_help())
print(args)  

dataset_str = args.dataset
method_str = args.model

# from config.json
import json
with open('config_data.json') as f:
    file_dict = json.load(f)

# ------ load data ------

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

para_lr_plain = lr_dic[dataset_str]
para_batch_size_plain = 64
para_is_stateful = False

para_decay = False
para_decay_step = 1000000

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

# epoch sample
para_val_epoch_num = int(0.05 * para_n_epoch)
para_test_epoch_num = 1

# ------ utility functions ------





def train_nn(num_dense, l2_dense, dropout_keep_prob, log_file, test_pickle, epoch_set):
    
    # log: epoch errors
    with open(log_file, "a") as text_file:
        text_file.write("\n num_dense: %d, keep_prob: %f, l2: %f \n"%(tmp_num_dense, tmp_keep_prob, tmp_l2))
    
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
        
        # fix the random seed to stabilize the network
        np.random.seed(1)
        tf.set_random_seed(1)
        
        # apply max_norm contraint only when dropout is used
        para_max_norm = maxnorm_dic[dataset_str] if dropout_keep_prob < 1.0 else 0.0
        
        if method_str == 'plain':
            
            print('\n\n ---', method_str, 
                  ' parameter: ',
                  ' l2-', l2_dense, 
                  ' dropout-', dropout_keep_prob, 
                  'maxnorm-', para_max_norm)
            
            reg = tsLSTM_plain(sess)
            
            reg.network_ini(para_lstm_dims_plain,
                            para_win_size,
                            para_input_dim,
                            para_lr_plain,
                            l2_dense,
                            para_max_norm,
                            para_bool_residual,
                            para_attention_plain,
                            l2_dense,
                            num_dense,
                            para_bool_regular_attention,
                            para_bool_regular_lstm,
                            para_bool_regular_dropout_output,
                            para_decay)
            
            para_batch_size = para_batch_size_plain
        
        # initialize training and inference
        reg.train_ini()
        reg.inference_ini()
        
        # data shuffling parameters
        total_cnt = np.shape(xtrain)[0]
        iter_per_epoch = int(total_cnt/para_batch_size) + 1
        total_idx = list(range(total_cnt))
        
        # model saver
        saver = tf.train.Saver()
        
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
                
                tmp_loss, tmp_err = reg.train_batch(batch_x, 
                                                    batch_y, 
                                                    dropout_keep_prob)
                loss_epoch += tmp_loss
                err_sum_epoch += tmp_err
                       
            # -- epoch-wise evaluation
            yh_val, val_rmse_epoch, val_mae_epoch, val_mape_epoch = reg.inference(xval, 
                                                                                  yval, 
                                                                                  1.0)
            ed_time_epoch = time.time()
            
            # test ?
            epoch_test_prediction.append(yh_val)
            epoch_error.append([epoch,
                                loss_epoch*1.0/iter_per_epoch,
                                sqrt(1.0*err_sum_epoch/total_cnt),
                                val_rmse_epoch,
                                val_mae_epoch,
                                val_mape_epoch])
            # epoch-wise 
            print("\n --- At epoch %d : \n    %s, %d "%(epoch, str(epoch_error[-1][1:]), ed_time_epoch - st_time_epoch))
                
            with open(log_file, "a") as text_file:
                text_file.write("%s\n"%(str(epoch_error[-1])[1:-1]))
                
            # save the model w.r.t. the epoch in epoch_sample
            if epoch in epoch_set:
                
                saver.save(sess, '../../ts_results/model/' + method_str + '-' + str(epoch))
                print("    [MODEL SAVED] \n")
        
        ed_time = time.time()
        
        print("Optimization Finished!") 
        
        # ---- return results
        
        return sorted(epoch_error, key = lambda x: x[3]), 1.0*(ed_time - st_time)/para_n_epoch

def log_train(text_env):    
    
    text_env.write("\n---- dataset: %s \n"%(dataset_str))
    text_env.write("dataset shape: %s \n"%(str(np.shape(xtrain))))
    text_env.write("method: %s, %s  \n"%(method_str, attention_dic[method_str]))
    text_env.write("plain layer size: %s \n"%(str(hidden_dim_dic[dataset_str])))
    text_env.write("lr: %s \n"%(str(lr_dic[dataset_str])))
    text_env.write("learnign rate decay : %s, %d \n"%(para_decay, para_decay_step))
    text_env.write("attention: %s \n"%(para_attention_plain))
    text_env.write("loss type: %s \n"%(para_loss_type))
    text_env.write("batch size: %s \n"%(str(para_batch_size_plain)))
    
    text_env.write("maximum norm constraint : %f \n"%(maxnorm_dic[dataset_str]))
    text_env.write("number of epoch : %d \n"%(para_n_epoch))
    text_env.write("regularization on LSTM weights : %s \n"%(para_bool_regular_lstm))
    text_env.write("regularization on attention : %s \n"%(para_bool_regular_attention))
    text_env.write("dropout before the outpout layer : %s \n"%(para_bool_regular_dropout_output))
    
    text_env.write("epoch num in validation : %s \n"%(para_val_epoch_num))
    text_env.write("epoch ensemble num in testing : %s \n\n"%(para_test_epoch_num))
    
    return

def log_val(text_env, best_hpara, epoch_set, best_val_err):
    
    text_env.write("\n best hyper parameters: %s %s \n"%(str(best_hpara), str(epoch_set)))
    text_env.write(" best validation errors: %s \n"%(str(best_val_err)))
    
    return

def log_test(text_env, errors):
    
    text_env.write("\n testing error: %s \n\n"%(errors))
    
    return

# ------ main process ------

'''
Log and dump files:

ts_rnn.txt: overall errors, all method, all set-up

log_method_dataset: epoch level training errors, method dataset wise

pred_pickle: only for MV-RNN, set-up wise

'''

if __name__ == '__main__':
    
    # log: overall erros, hyperparameter
    log_err_file = "../../ts_results/ts_plain.txt"
    with open(log_err_file, "a") as text_file:
        log_train(text_file)
        
    # log: epoch files
    log_epoch_file = "../../ts_results/log_" + method_str + "_" + dataset_str + ".txt"
    with open(log_epoch_file, "a") as text_file:
        log_train(text_file)
    
    # fix the random seed to reproduce the results
    np.random.seed(1)
    tf.set_random_seed(1)
    
    # ------ training and validation
    
    # grid search process
    hpara = []
    hpara_err = []
    
    # for tmp_lr in [0.001, 0.002, 0.005]
    for tmp_num_dense in [0, 1]:
        for tmp_keep_prob in [1.0, 0.8]:
            for tmp_l2 in [0.00001, 0.0001, 0.001, 0.01]:
                
                # pickle: predictions            
                pred_pickle = "../../ts_results/pred_"\
                              + str(dataset_str) + "_" \
                              + str(tmp_num_dense) + \
                              str(tmp_keep_prob) + \
                              str(tmp_l2) + "_"
                                
                # -- training
                error_epoch_log, epoch_time = train_nn(tmp_num_dense,
                                                       tmp_l2, 
                                                       tmp_keep_prob, 
                                                       log_epoch_file,
                                                       pred_pickle,
                                                       [])
                
                # error_epoch_log: [epoch, loss, train_rmse, val_rmse, val_mae, val_mape]
                hpara.append([tmp_num_dense, tmp_keep_prob, tmp_l2])
                hpara_err.append(error_epoch_log) 
                
                print('\n --- current running: ', tmp_num_dense, tmp_keep_prob, tmp_l2, error_epoch_log[0], '\n')
                
                # log: overall errors, hyperparameter set-up
                with open(log_err_file, "a") as text_file:
                    text_file.write("%f %f %f %s %s \n"%(tmp_num_dense, 
                                                          tmp_keep_prob, 
                                                          tmp_l2, 
                                                          str(error_epoch_log[0]), 
                                                          str(epoch_time)))
                    
    with open(log_err_file, "a") as text_file:
        text_file.write( "\n") 
    
    # ------ re-training
    
    best_hpara, epoch_sample, best_val_err = hyper_para_selection(hpara, 
                                                                  hpara_err, 
                                                                  para_val_epoch_num, 
                                                                  para_test_epoch_num)
    
    best_num_dense = best_hpara[0]
    best_keep_prob = best_hpara[1]
    best_l2 = best_hpara[2]
    
    # result record
    print('\n\n----- re-traning ------ \n')
    
    print('best hyper parameters: ', best_hpara, epoch_sample, '\n')
    print('best validation errors: ', best_val_err, '\n')
    
    with open(log_err_file, "a") as text_file:
        log_val(text_file, best_hpara, epoch_sample, best_val_err)
    
    import json
    with open('../../ts_results/hyper_para/' + dataset_str + '_' + method_str + '_' + para_attention_plain + '.json', 'w') as fp:
        
        tmp_hyper_para = {'num_dense':best_hpara[0],
                          'keep_prob':best_hpara[1],
                          'l2':best_hpara[2]
                         }
        
        json.dump(tmp_hyper_para, fp)
    
    # start the re-training
    error_epoch_log, epoch_time = train_nn(best_num_dense, 
                                           best_l2, 
                                           best_keep_prob, 
                                           log_epoch_file, 
                                           pred_pickle, 
                                           epoch_sample)
    
    # log: overall errors, performance for one hyperparameter set-up
    with open(log_err_file, "a") as text_file:
        text_file.write( "%f %f %f %s %s \n"%(best_num_dense, 
                                              best_keep_prob, 
                                              best_l2, 
                                              str(error_epoch_log[0]), 
                                              str(epoch_time)))
    # ------ testing
    
    print('\n\n----- testing ------ \n')
    
    yh, rmse, mae, mape = test_nn(epoch_sample, xval, yval, '../../ts_results/model/', method_str)
    print('\n\n testing errors: ', rmse, mae, mape, '\n\n')
    
    with open(log_err_file, "a") as text_file:
        log_test(text_file, [rmse, mae, mape])
    
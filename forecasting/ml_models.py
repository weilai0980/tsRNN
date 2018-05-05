import numpy as np   
import pandas as pd 

from pandas import *
from numpy import *
from scipy import *
 
import random

# machine leanring packages
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.ensemble import *
#GradientBoostingRegressor
from sklearn.ensemble import *
#RandomForestRegressor
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import xgboost as xgb

#from utils_keras import *
from utils_data_prep import *


# GBT

# https://www.analyticsvidhya.com/blog/2016/02/
# complete-guide-parameter-tuning-gradient-boosting-gbm-python/

#----Boosting parameters:
#   learnning rate: 0.05 - 0.2
#   n_estimators: 40-70

#----Tree parameters:
#   max_depth: 3-10
#   max_leaf_nodes
#   num_samples_split: 0.5-1% of total number 
#   min_samples_leaf
#   max_features

#   subsample: 0.8
#   min_weight_fraction_leaf

#----Order of tuning: max_depth and num_samples_split, min_samples_leaf, max_features



def gbt_n_estimatior(maxnum, X, Y, xtest, ytest, fix_lr, bool_clf ):
    
    tmpy = Y.reshape( (len(Y),) )
    score = []
    cnt = len(xtest)
    
    for trial_n in range(10,maxnum+1,10):
        
        if bool_clf == False:
            clf = GradientBoostingRegressor(n_estimators = trial_n,learning_rate = fix_lr)
        else:
            clf = GradientBoostingClassifier(n_estimators = trial_n,learning_rate = fix_lr)

        
        clf.fit( X, tmpy )
        
        pytest = clf.predict(xtest)

        if bool_clf == False:
            score.append((trial_n, sqrt(mean([( pytest[i]-ytest[i] )**2 for i in range(cnt) ]))) )
        else:
            score.append((trial_n, clf.score(xtest, ytest) ))
    
    return min(score, key = lambda x: x[1]), score


def gbt_tree_para( X, Y, xtest, ytest, depth_range, fix_lr, fix_n_est, bool_clf ):
    
    tmpy = Y.reshape( (len(Y),) )
    score = []
    
    cnt = len(xtest)
    
    for trial_depth in depth_range:
        
        if bool_clf == False:
            clf = GradientBoostingRegressor(n_estimators = fix_n_est,learning_rate = fix_lr,\
                                        max_depth = trial_depth )
        else:
            clf = GradientBoostingClassifier(n_estimators = fix_n_est,learning_rate = fix_lr,\
                                        max_depth = trial_depth )
            
        clf.fit( X, tmpy )
        
        pytest = clf.predict(xtest)
        
        if bool_clf == False:
            score.append((trial_depth, sqrt(mean([( pytest[i]-ytest[i] )**2 for i in range(cnt) ]))) )
        
        else:
            score.append((trial_depth, clf.score(xtest, ytest) ))
        
    return min(score, key = lambda x: x[1]), score
        
    
# XGBoosted

# https://www.analyticsvidhya.com/blog/2016/03/
#     complete-guide-parameter-tuning-xgboost-with-codes-python/

#----General Parameters

#   eta(learning rate): 0.05 - 0.3
#   number of rounds: 

#----Booster Parameters

#   max_depth 3-10
#   max_leaf_nodes
#   gamma: mininum loss reduction
#   min_child_weight: 1 by default

#   max_delta_step: not needed in general, for unbalance in logistic regression
#   subsample: 0.5-1
#   colsample_bytree: 0.5-1
#   colsample_bylevel: 
#   lambda: l2 regularization 
#   alpha: l1 regularization
#   scale_pos_weight: >>1, for high class imbalance

# Learning Task Parameters


def xgt_n_depth( lr, max_depth, max_round, xtrain, ytrain, xtest, ytest, bool_clf, num_class ):
    
    score = []
    xg_train = xgb.DMatrix(xtrain, label = ytrain)
    xg_test  = xgb.DMatrix(xtest,  label = ytest)

# setup parameters for xgboost
    param = {}
# use softmax multi-class classification

    if bool_clf == True:
        param['objective'] = 'multi:softmax'
        param['num_class'] = num_class
    else:
        param['objective'] = "reg:linear" 
#   'multi:softmax'
    
# scale weight of positive examples
    param['eta'] = lr
    param['max_depth'] = 0
    param['silent'] = 1
    param['nthread'] = 8
    
#     param['gamma']
    
    for depth_trial in range(2, max_depth):
        for num_round_trial in range(2, max_round):
            
            param['max_depth'] = depth_trial
            bst  = xgb.train( param, xg_train, num_round_trial )
            pred = bst.predict( xg_test )
            
            if bool_clf == True:
                tmplen = len(ytest)
                tmpcnt = 0.0
                for i in range(tmplen):
                    if ytest[i] == pred[i]:
                        tmpcnt +=1
                        
                tmp_accur = tmpcnt*1.0/tmplen
                    
            else:
                tmp_accur = sqrt(mean( [(pred[i] - ytest[i])**2 for i in range(len(ytest))] )) 
            
            score.append( (depth_trial, num_round_trial, tmp_accur) )
            
    return min(score, key = lambda x: x[2]), score


def xgt_l2( fix_lr, fix_depth, fix_round, xtrain, ytrain, xtest, ytest, l2_range, bool_clf, num_class ):
    
    score = []
    xg_train = xgb.DMatrix(xtrain, label = ytrain)
    xg_test  = xgb.DMatrix(xtest,  label = ytest)

# setup parameters for xgboost
    param = {}
# use softmax multi-class classification
    if bool_clf == True:
        param['objective'] = 'multi:softmax'
        param['num_class'] = num_class
    else:
        param['objective'] = "reg:linear" 

#   'multi:softmax'
    
# scale weight of positive examples
    param['eta'] = fix_lr
    param['max_depth'] = fix_depth
    param['silent'] = 1
    param['nthread'] = 8
    
    param['lambda'] = 0.0
#     param['alpha']
    
    
    for l2_trial in l2_range:
        
        param['lambda'] = l2_trial
        
        bst = xgb.train(param, xg_train, fix_round )
        pred = bst.predict( xg_test )
        
        if bool_clf == True:
            tmplen = len(ytest)
            tmpcnt = 0.0
            for i in range(tmplen):
                if ytest[i] == pred[i]:
                    tmpcnt +=1
                        
            tmp_accur = tmpcnt*1.0/tmplen
                    
        else:
            tmp_accur = sqrt(mean( [(pred[i] - ytest[i])**2 for i in range(len(ytest))] )) 
                    
        score.append( (l2_trial, tmp_accur) )
            
    return min(score, key = lambda x: x[1]), score

    
#  def xgt_l1 for very high dimensional features    
    
    
# ---- Random forest

# max_features:
# n_estimators
# max_depth

def rf_n_depth_estimatior(maxnum, maxdep, X, Y, xtest, ytest, bool_clf ):
        
    tmpy = Y
    score = []
    
    cnt = len(xtest)
        
    for n_trial in range(10,maxnum+1,10):
        for dep_trial in range(2, maxdep+1):
            
            if bool_clf == True:
                clf = RandomForestClassifier(n_estimators = n_trial, max_depth = dep_trial, max_features = "sqrt")
            else:
                clf = RandomForestRegressor(n_estimators = n_trial, max_depth = dep_trial, max_features = "sqrt")
            
            clf.fit( X, tmpy )
        
            pytest = clf.predict(xtest)
            
            if bool_clf == False:
                score.append( (n_trial, dep_trial, sqrt(mean([(pytest[i]-ytest[i])**2 for i in range(cnt) ]))) )
            else:
                score.append( (n_trial, dep_trial, clf.score(xtest, ytest)) )
            
#            score.append(\
#            (n_trial, dep_trial, sqrt(mean([( pytest[i]-ytest[i] )**2 for i in range(cnt) ]))) )
    
    return min(score, key = lambda x: x[2]), score


# ---- ElasticNet
from sklearn.linear_model import ElasticNet

def enet_alpha_l1(alpha_range, l1_range, xtrain, ytrain, xtest, ytest):
    
    print np.shape(ytest)
    
    res = []
    for i in alpha_range:
        for j in l1_range:
            
            enet = ElasticNet(alpha = i, l1_ratio = j)
            enet.fit(xtrain, ytrain)
            
            y_pred = enet.predict( xtest )
            
            res.append( (i,j, sqrt(mean([(ytest[k]-ytmp)**2 for k, ytmp in enumerate(y_pred)])) ) )
    
    return min(res, key = lambda x:x[2]), res
